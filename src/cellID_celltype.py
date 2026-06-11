"""Export cell ID to cell type mapping as CSV files for use in domain construction."""

import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Set base directory
base_dir = Path(
    "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/"
)
input = base_dir / "2026-03-27_analysis_run/adata_final_object"
output = base_dir / "muspan"


# Load .zarr
adata = ad.read_zarr(input / "adata_with_metadata.zarr")

# Extract cell ID and cluster labels
level_1_cell_id_to_type_df = adata.obs[["cell_id", "level 1"]].reset_index(drop=True)
level_2_cell_id_to_type_df = adata.obs[["cell_id", "level 2"]].reset_index(drop=True)
level_3_cell_id_to_type_df = adata.obs[["cell_id", "level 3"]].reset_index(drop=True)

# Export to CSV
level_1_cell_id_to_type_df.to_csv(
    output / "cell_id_to_cluster_labels_level1.csv", index=False
)
level_2_cell_id_to_type_df.to_csv(
    output / "cell_id_to_cluster_labels_level2.csv", index=False
)

level_3_cell_id_to_type_df.to_csv(
    output / "cell_id_to_cluster_labels_level3.csv", index=False
)
