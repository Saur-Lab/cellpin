import anndata as ad
import numpy as np
import pytest

from cellpin.dataset import scAnnDataset, stAnnDataset
from cellpin.pp import setup, setup_data

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def sc_adata():
    X = np.arange(1, 31, dtype=np.float32).reshape(2, 15)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene{i}" for i in range(15)]
    return adata


@pytest.fixture
def st_adata():
    X = np.ones((3, 5), dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene{i}" for i in range(5)]  # gene0..gene4 overlap with sc
    return adata


# ------------------------------------------------------------------ #
# Basic sanity (original tests)                                        #
# ------------------------------------------------------------------ #


def test_setup_data_returns_correct_types(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    assert isinstance(sc_ds, scAnnDataset)
    assert isinstance(st_ds, stAnnDataset)


def test_panel_genes_match(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    assert sc_ds.panel_genes == st_ds.panel_genes


def test_panel_genes_are_subset_of_spatial_genes(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    spatial_genes = set(st_adata.var_names.tolist())
    assert set(sc_ds.panel_genes).issubset(spatial_genes)


def test_no_overlap_raises():
    sc = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    sc.var_names = ["a", "b", "c"]
    st = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    st.var_names = ["x", "y"]
    with pytest.raises(ValueError, match="No overlapping genes"):
        setup_data(sc_adata=sc, st_adata=st)


# ------------------------------------------------------------------ #
# Duplicate gene detection                                             #
# ------------------------------------------------------------------ #


def test_duplicate_sc_genes_raises():
    sc = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    sc.var_names = pd_index_with_dupes(["a", "b", "a"])
    st = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    st.var_names = ["a", "b"]
    with pytest.raises(ValueError, match="sc_adata contains.*duplicate"):
        setup_data(sc_adata=sc, st_adata=st)


def test_duplicate_st_genes_raises():
    sc = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    sc.var_names = ["a", "b", "c"]
    st = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    st.var_names = pd_index_with_dupes(["a", "a"])
    with pytest.raises(ValueError, match="st_adata contains.*duplicate"):
        setup_data(sc_adata=sc, st_adata=st)


# ------------------------------------------------------------------ #
# Gene ordering invariant                                              #
# ------------------------------------------------------------------ #


def test_panel_order_sorted_by_sc_position():
    """panel_genes must follow sc_adata position order, not spatial order."""
    sc = ad.AnnData(X=np.arange(1, 7, dtype=np.float32).reshape(1, 6))
    sc.var_names = ["a", "b", "c", "d", "e", "f"]  # positions 0-5
    st = ad.AnnData(X=np.ones((1, 4), dtype=np.float32))
    st.var_names = ["d", "b", "f", "a"]  # intentionally shuffled
    sc_ds, st_ds = setup_data(sc_adata=sc, st_adata=st)
    # Expected: sorted by sc position → a(0), b(1), d(3), f(5)
    assert sc_ds.panel_genes == ["a", "b", "d", "f"]
    assert st_ds.panel_genes == ["a", "b", "d", "f"]


def test_panel_idx_is_strictly_ascending():
    """scAnnDataset.panel_idx must be ascending (boolean mask order guarantee)."""
    sc = ad.AnnData(X=np.arange(1, 7, dtype=np.float32).reshape(1, 6))
    sc.var_names = ["a", "b", "c", "d", "e", "f"]
    st = ad.AnnData(X=np.ones((1, 4), dtype=np.float32))
    st.var_names = ["d", "b", "f", "a"]
    sc_ds, _ = setup_data(sc_adata=sc, st_adata=st)
    idx = sc_ds.panel_idx
    assert len(idx) > 1
    assert np.all(np.diff(idx) > 0), f"panel_idx not ascending: {idx.tolist()}"


# ------------------------------------------------------------------ #
# VALUE alignment — the critical end-to-end check                     #
# ------------------------------------------------------------------ #


def test_panel_expr_values_correctly_aligned():
    """
    sc[0] = [1, 2, 3, 4, 5, 6] for genes [a, b, c, d, e, f]
    st[0] has genes [d, b, f, a] with values [40, 20, 60, 10]
      => d=40, b=20, f=60, a=10

    panel_genes (sc order): [a, b, d, f]

    sc panel_expr[0] should be [1, 2, 4, 6]   (a, b, d, f from sc)
    st full_expr[0] should be  [10, 20, 40, 60]  (a, b, d, f reordered from st)
    """
    sc = ad.AnnData(X=np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32))
    sc.var_names = ["a", "b", "c", "d", "e", "f"]

    st = ad.AnnData(X=np.array([[40, 20, 60, 10]], dtype=np.float32))
    st.var_names = ["d", "b", "f", "a"]

    sc_ds, st_ds = setup_data(sc_adata=sc, st_adata=st)

    assert sc_ds.panel_genes == ["a", "b", "d", "f"]
    assert st_ds.panel_genes == ["a", "b", "d", "f"]

    sc_panel = sc_ds[0]["panel_expr"].numpy()
    st_panel = st_ds[0]["full_expr"].numpy()

    np.testing.assert_array_equal(sc_panel, [1.0, 2.0, 4.0, 6.0], err_msg=f"sc panel_expr wrong: {sc_panel}")
    np.testing.assert_array_equal(st_panel, [10.0, 20.0, 40.0, 60.0], err_msg=f"st full_expr wrong: {st_panel}")


def test_panel_expr_gene_by_gene_alignment():
    """For every k, sc panel_expr[k] and st full_expr[k] refer to the same gene."""
    sc = ad.AnnData(X=np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32))
    sc.var_names = ["a", "b", "c", "d", "e", "f"]

    st = ad.AnnData(X=np.array([[40, 20, 60, 10]], dtype=np.float32))
    st.var_names = ["d", "b", "f", "a"]

    sc_ds, st_ds = setup_data(sc_adata=sc, st_adata=st)

    sc_panel = sc_ds[0]["panel_expr"].numpy()
    st_panel = st_ds[0]["full_expr"].numpy()
    sc_full = sc_ds[0]["full_expr"].numpy()

    # Known ground truth: gene → sc value, st value
    gene_to_sc = {"a": 1, "b": 2, "d": 4, "f": 6}
    gene_to_st = {"a": 10, "b": 20, "d": 40, "f": 60}

    for k, gene in enumerate(sc_ds.panel_genes):
        assert sc_panel[k] == gene_to_sc[gene], (
            f"sc panel_expr[{k}] = {sc_panel[k]}, expected {gene_to_sc[gene]} for gene '{gene}'"
        )
        assert st_panel[k] == gene_to_st[gene], (
            f"st full_expr[{k}] = {st_panel[k]}, expected {gene_to_st[gene]} for gene '{gene}'"
        )
        # Also verify sc panel value matches sc full_expr at the right position
        sc_idx = sc_ds.gene_names.tolist().index(gene)
        assert sc_panel[k] == sc_full[sc_idx], (
            f"sc panel_expr[{k}] ({sc_panel[k]}) != sc full_expr[{sc_idx}] ({sc_full[sc_idx]}) for gene '{gene}'"
        )


# ------------------------------------------------------------------ #
# Partial overlap / missing genes                                      #
# ------------------------------------------------------------------ #


def test_missing_st_genes_in_sc_does_not_raise(capsys):
    """Spatial genes absent from sc should be dropped with a warning, not raise."""
    sc = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    sc.var_names = ["a", "b", "c"]
    st = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    st.var_names = ["a", "b", "z"]  # "z" not in sc
    sc_ds, st_ds = setup_data(sc_adata=sc, st_adata=st)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "z" not in sc_ds.panel_genes
    assert "z" not in st_ds.panel_genes
    assert set(sc_ds.panel_genes) == {"a", "b"}


def test_panel_genes_also_subset_of_sc_genes(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    sc_all = set(sc_adata.var_names.tolist())
    assert set(sc_ds.panel_genes).issubset(sc_all)


# ------------------------------------------------------------------ #
# gene_symbols parameter                                               #
# ------------------------------------------------------------------ #


def test_with_gene_symbols():
    sc = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    sc.var["symbol"] = ["a", "b", "c"]
    st = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    st.var["symbol"] = ["b", "a"]
    sc_ds, st_ds = setup_data(sc_adata=sc, st_adata=st, gene_symbols="symbol")
    # panel_genes should be in sc order: a(0), b(1)
    assert sc_ds.panel_genes == ["a", "b"]
    assert st_ds.panel_genes == ["a", "b"]


# ------------------------------------------------------------------ #
# setup() alias                                                        #
# ------------------------------------------------------------------ #


def test_setup_alias_is_equivalent(sc_adata, st_adata):
    sc_ds1, st_ds1 = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    sc_ds2, st_ds2 = setup(sc_adata=sc_adata, st_adata=st_adata)
    assert sc_ds1.panel_genes == sc_ds2.panel_genes
    assert st_ds1.panel_genes == st_ds2.panel_genes
    assert isinstance(sc_ds2, scAnnDataset)
    assert isinstance(st_ds2, stAnnDataset)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def pd_index_with_dupes(names: list[str]):
    """Return a pandas Index that allows duplicate entries (for testing)."""
    import pandas as pd

    return pd.Index(names)
