#!/usr/bin/env python3
"""
run_cellpin.py
==============
Simple CellPin training and spatial imputation

Returns an anndata object containing both the original spatial data and the imputed values + CellPin embeddings.

Usage
-----
uv run python run_cellpin.py \
    --adata_path /path/to/scRNA.h5ad \
    --spatial_path /path/to/spatial.h5ad \
    --output_dir ./experiments/cellpin_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import yaml
from torch.utils.data import DataLoader
import cellpin
import cellpin.pp
from cellpin.models import CellPin
from cellpin.training import CorrelationCallback


#  Model config (YAML)

CONFIG_PATH_DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "cellpin_config.yaml"


def load_model_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser() if args.config else CONFIG_PATH_DEFAULT
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    model_config = load_model_config(config_path)
    print(f"Using model config: {config_path}")

    layer = args.layer or None

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading scRNA  : {args.adata_path}")
    adata_sc = ad.read_h5ad(args.adata_path)
    print(f"Loading spatial: {args.spatial_path}")
    adata_sp = ad.read_h5ad(args.spatial_path)
    print(f"sc: {adata_sc.shape}  |  spatial: {adata_sp.shape}")

    # ── Setup datasets for training ────────────────────────────────────────
    sc_dataset, st_dataset = cellpin.pp.setup_data(
        sc_adata=adata_sc,
        st_adata=adata_sp,
        gene_symbols=None,
        layer=layer,
    )

    # ── Create model ───────────────────────────────────────────────────────
    model = CellPin(sc_dataset=sc_dataset, config=model_config)

    shared = dict(
        precision=args.precision,
        accelerator="auto",
        devices=args.devices,
        strategy=args.strategy,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gradient_clip_val=0.5,
        early_stopping_patience=args.early_stopping_patience,
    )

    # ── Stage 1: pretrain ──────────────────────────────────────────────────
    print("\nStage 1: pretraining...")
    model.pretrain_model(
        dataset=sc_dataset,
        max_epochs=args.pretrain_epochs,
        output_dir=str(output_dir / "pretrain"),
        **shared,
    )

    # ── Stage 2: main training ─────────────────────────────────────────────
    print("\nStage 2: main training...")
    model.train_model(
        dataset=sc_dataset,
        freeze_pretrained=args.freeze_pretrained,
        decoder_warm_unfreeze_epoch=args.decoder_warm_unfreeze_epoch,
        max_epochs=args.train_epochs,
        output_dir=str(output_dir / "train"),
        **shared,
    )

    # ── Inference on spatial data ──────────────────────────────────────────
    print("\nRunning imputation on spatial data...")
    dl = DataLoader(
        st_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    adata_imputed = model.impute(dl, obs_adata=adata_sp, return_norm=False, nb_count_samples=20, return_int=False)

    out_path = output_dir / "cellpin_imputed.h5ad"
    adata_imputed.write_h5ad(str(out_path))
    print(f"Imputed AnnData saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CellPin: two-stage training + spatial imputation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adata_path",   required=True, help="scRNA AnnData (.h5ad)")
    p.add_argument("--spatial_path", required=True, help="Spatial AnnData (.h5ad)")
    p.add_argument("--output_dir",   default="./experiments/cellpin_run")
    p.add_argument("--config",       default=None, help=f"Path to model YAML config (default: {CONFIG_PATH_DEFAULT})")
    p.add_argument("--layer",        default="counts", help="Expression layer (empty → .X)")

    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--train_epochs",    type=int, default=60)
    p.add_argument("--batch_size",      type=int, default=256)
    p.add_argument("--num_workers",     type=int, default=8)
    p.add_argument("--early_stopping_patience", type=int, default=12)

    p.add_argument("--freeze_pretrained",             action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decoder_warm_unfreeze_epoch",   type=int, default=25)

    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--devices",   type=int, nargs="+", default=[0])
    p.add_argument("--strategy",  default="auto")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
