from types import SimpleNamespace

import anndata as ad
import numpy as np

from cellpin.tl import label_transfer


def test_label_transfer_refits_on_full_reference_for_spatial_prediction():
    sc_adata = ad.AnnData(X=np.zeros((8, 1), dtype=np.float32))
    sc_adata.obs["cell_type"] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    sc_adata.obsm["X_cellpin"] = np.array(
        [[0.10], [0.20], [0.30], [0.40], [10.0], [10.1], [0.0], [10.2]],
        dtype=np.float32,
    )

    sp_adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    sp_adata.obsm["X_cellpin"] = np.array([[0.0]], dtype=np.float32)

    model = SimpleNamespace(panel_gene_names=[])

    accuracy, result = label_transfer(
        model,
        sc_adata,
        "cell_type",
        sp_adata,
        k=1,
        test_size=0.25,
        random_state=42,
    )

    assert accuracy == 0.5
    assert result.obs["cellpin_annotation"].astype(str).iloc[0] == "B"
