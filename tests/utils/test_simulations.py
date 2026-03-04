import unittest

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.datasets import make_blobs

from locat.locat_condensed import LOCAT


def create_anndata(matrix, cell_names=None, gene_names=None):
    adata = ad.AnnData(matrix)
    adata.obs_names = cell_names or [f"Cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = gene_names or [f"Gene_{i}" for i in range(adata.n_vars)]
    return adata


def simulate_blob_data(n_samples=500, n_tests=20, n_total=50):
    coords, _, centers = make_blobs(
        n_samples=[n_samples],
        n_features=2,
        centers=None,
        return_centers=True,
        random_state=0,
        cluster_std=[1.0],
    )

    rng = np.random.default_rng(0)
    radius0 = 0.5
    fractions_in = np.linspace(1.0, 0.5, n_tests)

    dists = np.sqrt(np.sum((coords - centers[0, :]) ** 2, axis=1))
    in_region = np.flatnonzero(dists < radius0)
    out_region = np.flatnonzero(dists >= radius0)

    genes = np.zeros((coords.shape[0], n_tests), dtype=np.uint8)

    for i, frac_in in enumerate(fractions_in):
        n_in = int(np.round(frac_in * n_total))
        n_out = n_total - n_in

        n_in = min(n_in, len(in_region))
        n_out = min(n_out, len(out_region))

        cur_total = n_in + n_out
        if cur_total < n_total:
            need = n_total - cur_total
            room_in = len(in_region) - n_in
            add_in = min(need, room_in)
            n_in += add_in
            need -= add_in

            room_out = len(out_region) - n_out
            add_out = min(need, room_out)
            n_out += add_out

        idx_in = (
            rng.choice(in_region, n_in, replace=False) if n_in > 0 else np.array([], dtype=int)
        )
        idx_out = (
            rng.choice(out_region, n_out, replace=False)
            if n_out > 0
            else np.array([], dtype=int)
        )

        pos_idx = np.concatenate([idx_in, idx_out])
        genes[pos_idx, i] = 1

    adata = create_anndata(genes.astype(np.float64))
    adata.obsm["coords"] = coords.astype(np.float64)
    sc.pp.neighbors(adata, use_rep="coords", n_neighbors=30)
    return adata


class SimulationTestCase(unittest.TestCase):
    def test_simulated_data_significant(self):
        np.random.seed(0)
        adata = simulate_blob_data(n_samples=500, n_tests=20, n_total=50)
        gene_name = "Gene_0"

        locat = LOCAT(
            adata,
            adata.obsm["coords"],
            k=10,
            show_progress=False,
            knn=adata.obsp["connectivities"],
        )
        res = locat.gmm_scan_new(genes=[gene_name])
        gene_0 = res[gene_name]

        self.assertAlmostEqual(-1.512298, gene_0["bic"], places=4)
        self.assertGreaterEqual(gene_0["zscore"], 0.0)
        self.assertAlmostEqual(1.0, gene_0["sens_score"], places=6)
        self.assertLess(gene_0["depletion_pval"], 1e-6)
        self.assertAlmostEqual(1e-15, gene_0["concentration_pval"], places=14)
        self.assertLess(gene_0["pval"], 1e-2)
        self.assertEqual(1, gene_0["K_components"])
        self.assertEqual(50, int(gene_0["sample_size"]))


if __name__ == "__main__":
    unittest.main()
