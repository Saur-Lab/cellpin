"""Atlas-match → per-epoch finetune UMAP movie.

Workflow
--------
1. Load & (optionally) subsample scRNA + spatial data.
2. Train atlas-matching network (match_emb) on scRNA.
3. Embed spatial → atlas space; run label_transfer to get a FIXED annotation.
4. Fit UMAP once on combined sc-atlas + spatial embeddings.
5. Save frame 0 (after match_emb, before finetune).
6. finetune_spatial: after every validation epoch a callback re-embeds spatial,
   transforms through the frozen UMAP, and saves the next frame.
7. Concatenate all frames into an mp4 movie with ffmpeg.

Usage
-----
    # Quick test (10k sc + 10k sp, fewer epochs)
    python run_atlas_finetune_movie.py --test

    # Full run
    python run_atlas_finetune_movie.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
import umap as umap_lib

import cellpin
from cellpin.dataset import scAnnDataset


# ── Paths ─────────────────────────────────────────────────────────────────────
SC_PATH = (
    "/mnt/storage/philipp/PP_FlexResource/public/core/"
    "Core_annotated_v2_with_Level_5_harmonized.h5ad"
)
SP_PATH = (
    "/mnt/storage/philipp/PP_FlexResource/public/spatial/segmented/"
    "cellpose/panel_77ATW8/25337_segmented.h5ad"
)

LAYER         = "counts"
ATLAS_EMB_KEY = "x_scVI_1"

# Known ffmpeg locations (checked in order when ffmpeg is not on PATH)
_FFMPEG_CANDIDATES = [
    "ffmpeg",
    "/home/philipp.putze/micromamba/pkgs/https/conda.anaconda.org/conda-forge/"
    "linux-64/ffmpeg-8.1.2-gpl_h1bf8424_901/bin/ffmpeg",
    "/home/philipp.putze/micromamba/envs/cellcharter-env/bin/ffmpeg",
]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="Subsample to 10k sc + 10k sp; use fewer epochs")
    p.add_argument("--out_dir", default="./atlas_finetune_movie",
                   help="Output directory (default: ./atlas_finetune_movie)")
    p.add_argument("--cell_type_col", default="Level_4",
                   help="obs column for cell type labels (default: Level_4)")
    p.add_argument("--match_epochs",  type=int, default=None)
    p.add_argument("--finetune_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--subsample_sc", type=int, default=None)
    p.add_argument("--subsample_sp", type=int, default=None)
    return p.parse_args()


def _find_ffmpeg() -> str | None:
    for candidate in _FFMPEG_CANDIDATES:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def _make_movie(frames_dir: Path, out_dir: Path) -> None:
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    n_frames = len(frame_files)
    print(f"\nAssembling movie from {n_frames} frames...")

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        mp4_path = out_dir / "finetune_movie.mp4"
        cmd = (
            f'"{ffmpeg}" -y -framerate 2 '
            f'-pattern_type glob -i "{frames_dir}/frame_*.png" '
            f'-c:v libx264 -pix_fmt yuv420p '
            f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            f'"{mp4_path}"'
        )
        ret = subprocess.call(cmd, shell=True)
        if ret == 0:
            print(f"Movie (mp4) saved → {mp4_path}")
            return
        print(f"ffmpeg failed (exit {ret}), falling back to GIF...")

    # Fallback: write animated GIF with imageio (no external binary needed)
    import imageio.v2 as iio
    gif_path = out_dir / "finetune_movie.gif"
    frames = [iio.imread(str(f)) for f in frame_files]
    iio.mimwrite(str(gif_path), frames, duration=0.5)
    print(f"Movie (GIF) saved → {gif_path}")


# ── Plotting ──────────────────────────────────────────────────────────────────
def _build_colormap(categories: list[str]) -> dict[str, str]:
    # Use scanpy's default palette; fall back to tab20 for > 102 types
    palette = list(sc.pl.palettes.default_102)
    if len(categories) > len(palette):
        palette = [mcolors.to_hex(c) for c in plt.cm.tab20.colors] * 10
    return {ct: palette[i % len(palette)] for i, ct in enumerate(categories)}


def save_frame(
    sp_umap: np.ndarray,
    annotation,
    sc_umap: np.ndarray,
    color_map: dict[str, str],
    categories: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    # sc background (gray, semi-transparent)
    ax.scatter(
        sc_umap[:, 0], sc_umap[:, 1],
        s=1, c="lightgray", alpha=0.25, rasterized=True, label="_sc",
    )

    # spatial cells colored by the fixed annotation
    for ct in categories:
        mask = np.asarray(annotation == ct)
        if mask.sum() == 0:
            continue
        ax.scatter(
            sp_umap[mask, 0], sp_umap[mask, 1],
            s=5, c=color_map[ct], alpha=0.7, rasterized=True, label=ct,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[ct], markersize=6, label=ct)
        for ct in categories
        if np.asarray(annotation == ct).sum() > 0
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=6, ncol=max(1, len(handles) // 30))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [frame] Saved → {out_path.name}")


# ── Per-epoch callback ────────────────────────────────────────────────────────
class PerEpochUMAPCallback(pl.Callback):
    """Re-embed spatial after every finetune epoch and save a UMAP frame.

    Hooks at on_validation_epoch_end so that EMA weights (swapped in at
    on_validation_start by EMACallback) are active during embed_atlas.
    """

    def __init__(
        self,
        sp_dl: torch.utils.data.DataLoader,
        reducer: umap_lib.UMAP,
        sc_umap: np.ndarray,
        fixed_annotation,
        color_map: dict[str, str],
        categories: list[str],
        frames_dir: Path,
    ) -> None:
        super().__init__()
        self.sp_dl = sp_dl
        self.reducer = reducer
        self.sc_umap = sc_umap
        self.fixed_annotation = fixed_annotation
        self.color_map = color_map
        self.categories = categories
        self.frames_dir = frames_dir
        self._finetune_epoch = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return  # skip pre-training sanity validation

        self._finetune_epoch += 1
        frame_idx = self._finetune_epoch  # frame 0 reserved for post-match_emb

        sp_emb = pl_module.embed_atlas(self.sp_dl)
        sp_umap = self.reducer.transform(sp_emb)

        save_frame(
            sp_umap=sp_umap,
            annotation=self.fixed_annotation,
            sc_umap=self.sc_umap,
            color_map=self.color_map,
            categories=self.categories,
            title=f"Spatial → Atlas  |  fine-tune epoch {self._finetune_epoch}",
            out_path=self.frames_dir / f"frame_{frame_idx:03d}.png",
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    cell_type_col = args.cell_type_col

    # Resolve hyper-parameters (test mode overrides)
    if args.test:
        subsample_sc    = args.subsample_sc   or 10_000
        subsample_sp    = args.subsample_sp   or 10_000
        match_epochs    = args.match_epochs   or 10
        finetune_epochs = args.finetune_epochs or 5
        batch_size      = args.batch_size     or 512
    else:
        subsample_sc    = args.subsample_sc   or None
        subsample_sp    = args.subsample_sp   or None
        match_epochs    = args.match_epochs   or 60
        finetune_epochs = args.finetune_epochs or 30
        batch_size      = args.batch_size     or 1024

    out_dir    = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    sc_adata = sc.read_h5ad(SC_PATH)
    sp_adata = sc.read_h5ad(SP_PATH)

    print(f"  scRNA  : {sc_adata.n_obs:,} × {sc_adata.n_vars:,}")
    print(f"  Spatial: {sp_adata.n_obs:,} × {sp_adata.n_vars:,}")
    print(f"  Atlas embedding dim: {sc_adata.obsm[ATLAS_EMB_KEY].shape[1]}")

    if subsample_sc and sc_adata.n_obs > subsample_sc:
        sc.pp.subsample(sc_adata, n_obs=subsample_sc, random_state=42)
        print(f"  Subsampled scRNA → {sc_adata.n_obs:,}")

    if subsample_sp and sp_adata.n_obs > subsample_sp:
        sc.pp.subsample(sp_adata, n_obs=subsample_sp, random_state=42)
        print(f"  Subsampled spatial → {sp_adata.n_obs:,}")

    # Drop cell types with < 2 cells so label_transfer stratified split doesn't crash.
    # This only matters after heavy subsampling; no effect on full data.
    ct_counts = sc_adata.obs[cell_type_col].value_counts()
    keep_types = ct_counts[ct_counts >= 2].index
    n_before = sc_adata.n_obs
    sc_adata = sc_adata[sc_adata.obs[cell_type_col].isin(keep_types)].copy()
    if sc_adata.n_obs < n_before:
        print(f"  Dropped {n_before - sc_adata.n_obs} sc cells with singleton cell types")

    sp_adata.layers["counts"] = sp_adata.X

    # ── 2. Setup datasets ─────────────────────────────────────────────────────
    print("Setting up datasets...")
    sc_dataset, sp_dataset = cellpin.pp.setup_data(sc_adata, sp_adata, layer=LAYER)

    # ── 3. Build model ────────────────────────────────────────────────────────
    config = {
        "atlas_hidden": 1024,
        "atlas_blocks": 8,
        "atlas_expansion": 2.0,
        "atlas_dropout": 0.1,
        "atlas_drop_path_rate": 0.1,
        "poisson_resample_rate": 0.4,
        "spatial_resample_rate": 0.85,
        "panel_mixup_alpha": 0.3,
        "atlas_distill_weight": 1.0,
        "atlas_consistency_weight": 1.0,
        "atlas_cos_weight": 0.1,
        "atlas_aug_warmup_frac": 0.10,
        "atlas_lr_warmup_epochs": 5,
        "atlas_ema_decay": 0.999,
    }
    model = cellpin.CellPin(sc_dataset, config=config)

    # ── 4. Train atlas-matching network ───────────────────────────────────────
    print(f"\nTraining atlas-matching network ({match_epochs} max epochs)...")
    model.match_emb(
        sc_dataset,
        emb_key=ATLAS_EMB_KEY,
        train_epochs=match_epochs,
        batch_size=batch_size,
        early_stopping_patience=20,
        checkpoint_monitor="val_knn_overlap",
        early_stopping_mode="max",
        accelerator="gpu",
        devices=1,
    )

    # ── 5. Embed spatial + label transfer (FIXED annotation) ──────────────────
    sp_dl = torch.utils.data.DataLoader(
        sp_dataset, batch_size=512, shuffle=False, num_workers=4
    )

    print("\nEmbedding spatial cells...")
    sp_adata.obsm["X_cellpin_match"] = model.embed_atlas(sp_dl)

    acc, sp_adata = model.tl.label_transfer(sc_adata, cell_type_col, sp_adata)
    print(f"match_emb (pre-finetune) → kNN accuracy on held-out scRNA: {acc:.4f}")

    # Freeze annotation — never updated during finetune
    fixed_annotation = sp_adata.obs["cellpin_annotation"].copy()
    categories = list(fixed_annotation.cat.categories)
    color_map  = _build_colormap(categories)
    print(f"Fixed annotation: {len(categories)} cell types, {sp_adata.n_obs:,} spatial cells")

    # ── 6. Fit UMAP once on sc-atlas + spatial embeddings ────────────────────
    # Use the ground-truth scVI atlas positions for sc (reference space);
    # spatial positions are from atlas_net (trained to reproduce scVI space).
    sc_ref_emb = np.asarray(sc_adata.obsm[ATLAS_EMB_KEY], dtype=np.float32)
    sp_emb_0   = np.asarray(sp_adata.obsm["X_cellpin_match"], dtype=np.float32)

    print(f"\nFitting UMAP on {sc_ref_emb.shape[0]:,} sc + {sp_emb_0.shape[0]:,} sp cells...")
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        random_state=42,
        low_memory=True,
    )
    combined_umap = reducer.fit_transform(
        np.concatenate([sc_ref_emb, sp_emb_0], axis=0)
    )
    sc_umap   = combined_umap[: len(sc_ref_emb)]
    sp_umap_0 = combined_umap[len(sc_ref_emb) :]

    # ── Frame 0: after match_emb, before fine-tune ────────────────────────────
    print("\nSaving frame 0 (post-match_emb)...")
    save_frame(
        sp_umap=sp_umap_0,
        annotation=fixed_annotation,
        sc_umap=sc_umap,
        color_map=color_map,
        categories=categories,
        title="Spatial → Atlas  |  after match_emb (before fine-tune)",
        out_path=frames_dir / "frame_000.png",
    )

    # ── 7. Fine-tune with per-epoch UMAP frames ───────────────────────────────
    umap_callback = PerEpochUMAPCallback(
        sp_dl=sp_dl,
        reducer=reducer,
        sc_umap=sc_umap,
        fixed_annotation=fixed_annotation,
        color_map=color_map,
        categories=categories,
        frames_dir=frames_dir,
    )

    print(f"\nFine-tuning ({finetune_epochs} max epochs)...")
    model.finetune_spatial(
        sc_dataset,
        sp_dataset,
        sc_type_labels=sc_adata.obs[cell_type_col].values,
        sp_type_labels=np.asarray(fixed_annotation),
        train_epochs=finetune_epochs,
        batch_size=batch_size * 2,
        early_stopping_patience=10,
        accelerator="gpu",
        devices=1,
        custom_callbacks=[umap_callback],
    )

    # ── 8. Assemble movie ─────────────────────────────────────────────────────
    _make_movie(frames_dir, out_dir)


if __name__ == "__main__":
    main()
