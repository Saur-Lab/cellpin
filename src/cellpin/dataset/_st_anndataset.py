from __future__ import annotations

import anndata as ad
import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset


class stAnnDataset(Dataset):
    """Spatial AnnData dataset aligned to a panel gene list.

    Spatial dataset that returns expression aligned to an explicit ordered panel
    gene list.

    Invariant:
      - self.panel_genes defines the order of expr returned by __getitem__
      - expr[k] corresponds to self.panel_genes[k]

    local_l_mean / local_l_var are computed from the spatial data itself so
    that the library encoder prior is consistent at inference time.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        panel_genes: list[str],
        layer: str | None = None,
        gene_symbols: str | None = None,
    ) -> None:
        self.adata = adata
        self.layer = layer

        st_gene_names = (
            self.adata.var[gene_symbols].astype(str).to_numpy()
            if gene_symbols is not None
            else self.adata.var_names.astype(str).to_numpy()
        )

        st_map: dict[str, int] = {}
        for j, g in enumerate(st_gene_names.tolist()):
            if g not in st_map:
                st_map[g] = j

        missing = [g for g in panel_genes if g not in st_map]
        if missing:
            raise ValueError(
                f"stAnnDataset got panel_genes not present in spatial data (n={len(missing)}). Examples: {missing[:10]}"
            )

        col_idx = np.array([st_map[g] for g in panel_genes], dtype=np.int64)
        self.adata = self.adata[:, col_idx]
        self.panel_genes = list(panel_genes)
        self.X = self.adata.layers[self.layer] if self.layer is not None else self.adata.X
        self.is_sparse = issparse(self.X)

        if self.is_sparse:
            lib_sizes = np.array(self.X.sum(axis=1)).ravel()
        else:
            lib_sizes = np.array(self.X).sum(axis=1)

        log_lib_sizes = np.log1p(lib_sizes.astype(np.float64))
        self._local_l_mean = torch.tensor([float(log_lib_sizes.mean())], dtype=torch.float32)
        self._local_l_var = torch.tensor([float(log_lib_sizes.var() + 1e-6)], dtype=torch.float32)

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.is_sparse:
            expr = self.X.getrow(idx).toarray().ravel()
        else:
            expr = self.X[idx, :]

        expr = torch.tensor(expr, dtype=torch.float32)

        return {
            "full_expr": expr,
            "panel_expr": expr,  # spatial only has panel genes
            "local_l_mean": self._local_l_mean,
            "local_l_var": self._local_l_var,
        }
