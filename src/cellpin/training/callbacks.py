import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from pytorch_lightning.utilities import rank_zero_only


def _iter_dataloader(dataloaders):
    if dataloaders is None:
        return None
    if isinstance(dataloaders, (list, tuple)):
        if len(dataloaders) == 0:
            return None
        return dataloaders[0]
    return dataloaders


class CorrelationCallback(pl.Callback):
    def __init__(
        self,
        run_every_n_epochs: int = 5,
        num_samples: int = 1000,
        verbose: bool = True,
        no_panel_mask=None,
    ):
        super().__init__()
        self.run_every_n_epochs = run_every_n_epochs
        self.num_samples = num_samples
        self.verbose = verbose
        # Boolean mask (numpy array or torch.Tensor) of shape (n_genes,).
        # True = non-panel gene. When set, correlation is restricted to those genes.
        if no_panel_mask is not None:
            if isinstance(no_panel_mask, torch.Tensor):
                no_panel_mask = no_panel_mask.cpu().numpy()
            self.no_panel_mask = np.asarray(no_panel_mask, dtype=bool)
        else:
            self.no_panel_mask = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.run_every_n_epochs != 0:
            return

        val_dataloader = _iter_dataloader(trainer.val_dataloaders)
        if val_dataloader is None:
            if self.verbose:
                self._print("No validation dataloader available for correlation computation")
            return

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        all_predictions = []
        all_ground_truth = []
        all_head_losses = []
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if total >= self.num_samples:
                    break
                x_full = batch["full_expr"].to(device)
                x_panel = batch["panel_expr"].to(device)
                batch_index = batch.get("batch_index", None)
                if batch_index is not None:
                    batch_index = batch_index.to(device)

                out_panel = pl_module.vae.inference(
                    x_full,
                    x_panel=x_panel,
                    batch_index=batch_index,
                )
                pred = torch.clamp(out_panel["px_rate"], min=0)

                # Single reconstruction head for scVI-style decoder.
                head_losses = [F.mse_loss(pred, x_full).detach().cpu().item()]

                all_predictions.append(pred.detach().cpu().numpy())
                all_ground_truth.append(x_full.detach().cpu().numpy())
                all_head_losses.append(head_losses)
                total += x_panel.shape[0]

        if len(all_predictions) == 0:
            if self.verbose:
                self._print("⚠️  No samples available for correlation computation")
            if was_training:
                pl_module.train()
            return

        predictions = np.concatenate(all_predictions, axis=0)[: self.num_samples]
        ground_truth = np.concatenate(all_ground_truth, axis=0)[: self.num_samples]

        if self.no_panel_mask is not None:
            predictions = predictions[:, self.no_panel_mask]
            ground_truth = ground_truth[:, self.no_panel_mask]

        correlations = self._compute_gene_correlations(predictions, ground_truth)

        mean_corr = np.nanmean(correlations)
        median_corr = np.nanmedian(correlations)
        std_corr = np.nanstd(correlations)
        min_corr = np.nanmin(correlations)
        max_corr = np.nanmax(correlations)

        valid_genes = int(np.sum(~np.isnan(correlations)))
        total_genes = int(len(correlations))

        mean_head_losses = np.mean(np.asarray(all_head_losses, dtype=np.float64), axis=0)
        per_head_metrics = {
            f"reconstruction/head_{i + 1}_mse": float(v)
            for i, v in enumerate(mean_head_losses.tolist())
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "correlation/mean_pearson": float(mean_corr),
                    "correlation/median_pearson": float(median_corr),
                    "correlation/std_pearson": float(std_corr),
                    "correlation/min_pearson": float(min_corr),
                    "correlation/max_pearson": float(max_corr),
                    "correlation/valid_genes": float(valid_genes),
                    **per_head_metrics,
                },
                step=trainer.global_step,
            )

        if self.verbose:
            gene_scope = "non-panel genes only" if self.no_panel_mask is not None else "all genes"
            self._print("\n" + "=" * 70)
            self._print(f"CORRELATION ANALYSIS (Epoch {trainer.current_epoch}) [{gene_scope}]")
            self._print("=" * 70)
            self._print(f"Gene-wise Pearson Correlation (n={predictions.shape[0]} cells):")
            self._print(f"   Mean:   {mean_corr:.4f}")
            self._print(f"   Median: {median_corr:.4f}")
            self._print(f"   Std:    {std_corr:.4f}")
            self._print(f"   Range:  [{min_corr:.4f}, {max_corr:.4f}]")
            self._print(f"   Valid:  {valid_genes}/{total_genes} genes")
            self._print("Reconstruction MSE:")
            for i, v in enumerate(mean_head_losses.tolist(), start=1):
                self._print(f"   Head {i}: {v:.6f}")
            self._print("=" * 70 + "\n")

        if was_training:
            pl_module.train()
        
    @rank_zero_only
    def _print(self, msg: str):
        print(msg)

    def _compute_gene_correlations(self, predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        if issparse(predictions):
            predictions = predictions.toarray()
        if issparse(ground_truth):
            ground_truth = ground_truth.toarray()

        _, n_genes = predictions.shape
        correlations = np.zeros(n_genes, dtype=np.float64)

        for gene_idx in range(n_genes):
            pred_gene = predictions[:, gene_idx]
            true_gene = ground_truth[:, gene_idx]

            if pred_gene.std() > 0 and true_gene.std() > 0:
                correlations[gene_idx] = np.corrcoef(pred_gene, true_gene)[0, 1]
            else:
                correlations[gene_idx] = np.nan

        return correlations
    
