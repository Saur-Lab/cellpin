from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset


class scAnnDataset(Dataset):
    """scRNA-seq AnnData dataset wrapper. Returned by ``cellpin.pp.setup_data()``.

    Outputs (per observation):
        - full_expr:     full expression row (all genes, stable order)
        - panel_expr:    full_expr restricted to panel genes
        - no_panel_expr: full_expr restricted to non-panel genes
        - local_l_mean:  dataset-level mean of log-library size (1,)
        - local_l_var:   dataset-level variance of log-library size (1,)
        - batch_index:   integer batch label (only when batch_key is set)

    Args:
        adata: scRNA-seq AnnData object.
        layer: Expression layer to read. Must be raw counts. When ``None``,
            ``.X`` is used.
        gene_symbols: ``var`` column name for alternative gene identifiers
            (same semantics as in ``setup_data``).
        panel: Ordered list of panel gene names. Internally, panel gene order
            is fixed to ascending position in ``adata`` (boolean-mask order),
            so ``panel_expr[k]`` always corresponds to genes in that order,
            not the order of the ``panel`` argument.
        batch_key: ``obs`` column for integer batch labels. When ``None``,
            batch conditioning is off.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        layer: str | None = None,
        gene_symbols: str | None = None,
        panel: list[str] | None = None,
        batch_key: str | None = None,
    ) -> None:
        self.adata = adata
        self.layer = layer

        if gene_symbols is not None:
            gene_names = self.adata.var[gene_symbols].astype(str).to_numpy()
        else:
            gene_names = self.adata.var_names.to_numpy()

        self.X = self.adata.layers[self.layer] if self.layer is not None else self.adata.X
        self.is_sparse = issparse(self.X)
        n_genes = self.X.shape[1]

        if panel is not None and len(panel) > 0:
            gene_to_idx = {g: i for i, g in enumerate(gene_names)}
            missing_panel = [g for g in panel if g not in gene_to_idx]
            if missing_panel:
                warnings.warn(
                    f"scAnnDataset: {len(missing_panel)} panel gene(s) not found in adata "
                    f"and will be ignored: {missing_panel[:10]}" + (" ..." if len(missing_panel) > 10 else ""),
                    UserWarning,
                    stacklevel=2,
                )
            panel_idx = [gene_to_idx[g] for g in panel if g in gene_to_idx]
            panel_idx = np.asarray(panel_idx, dtype=np.int64)
            if panel_idx.size > 1:
                _, first_pos = np.unique(panel_idx, return_index=True)
                panel_idx = panel_idx[np.sort(first_pos)]
        else:
            panel_idx = np.empty((0,), dtype=np.int64)

        panel_mask_np = np.zeros(n_genes, dtype=bool)
        if panel_idx.size:
            panel_mask_np[panel_idx] = True

        self._panel_mask = torch.from_numpy(panel_mask_np)
        self._no_panel_mask = ~self._panel_mask
        self.gene_names = gene_names
        self.panel_idx = panel_idx
        self.panel_genes = [gene_names[i] for i in panel_idx.tolist()]

        # Batch conditioning
        self.n_batch: int = 0
        self._batch_indices: torch.Tensor | None = None
        if batch_key is not None:
            if batch_key not in adata.obs.columns:
                warnings.warn(
                    f"scAnnDataset: batch_key='{batch_key}' not found in adata.obs; batch conditioning disabled.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                cats = adata.obs[batch_key].astype("category")
                self.n_batch = len(cats.cat.categories)
                self._batch_indices = torch.tensor(cats.cat.codes.to_numpy().astype(np.int64), dtype=torch.long)

        if self.is_sparse:
            lib_sizes = np.array(self.X.sum(axis=1)).ravel()
        else:
            lib_sizes = self.X.sum(axis=1)

        log_lib_sizes = np.log1p(lib_sizes.astype(np.float64))
        self._local_l_mean = torch.tensor([float(log_lib_sizes.mean())], dtype=torch.float32)
        self._local_l_var = torch.tensor([float(log_lib_sizes.var() + 1e-6)], dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.is_sparse:
            full_expr = self.X.getrow(idx).toarray().ravel()
        else:
            full_expr = self.X[idx, :]

        full_expr = torch.tensor(full_expr, dtype=torch.float32)
        panel_expr = full_expr[self._panel_mask]
        no_panel_expr = full_expr[self._no_panel_mask]

        out = {
            "full_expr": full_expr,
            "panel_expr": panel_expr,
            "no_panel_expr": no_panel_expr,
            "local_l_mean": self._local_l_mean,
            "local_l_var": self._local_l_var,
        }
        if self._batch_indices is not None:
            out["batch_index"] = self._batch_indices[idx]
        return out
