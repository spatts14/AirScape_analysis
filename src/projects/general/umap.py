"""Generate misc plots."""

from pathlib import Path

import anndata as ad
import scanpy as sc
import seaborn as sns

from utils.seed_everything import seed_everything

# Set random seed for reproducibility
seed_everything(19960915)

# Set directories
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)

fig_dir = dir / "project_analysis/general/umap"
fig_dir.mkdir(parents=True, exist_ok=True)


# Configure scanpy to save figures in our custom directory
sc.settings.figdir = fig_dir

# Set colors
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
color_palette_level_1 = sns.color_palette("hls", 12)

# Load data
print(f"Loading data from {dir / 'adata_final_object/adata_with_metadata.zarr'}...")
adata = ad.read_zarr(dir / "adata_final_object/adata_with_metadata.zarr")

print("Data loaded successfully.")

# Set variables
color_list = adata.obs.columns.tolist()

# Remove columns that will cause an error when plotting
color_list = [
    "time_point",
    "condition",
    "lung_location",
    "biopsy_type",
    "diagnosis",
    "treatment_arm",
    "age",
    "sex",
    "smoking_status",
    "batch",
    "level_1",
    "level_2",
]

print("Plotting UMAP...")
for color in color_list:
    if color not in adata.obs.columns:
        print(f"Warning: {color} not found in adata.obs. Skipping this variable.")
        continue
    sc.pl.umap(
        adata,
        color=color,
        cmap=cmap,
        wspace=0.4,
        show=False,
        frameon=False,
        save=f"_{color}.png",
    )

print("UMAP plotted and saved successfully.")
