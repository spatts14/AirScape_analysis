"""Plot PCA and spatial distribution of clusters for the full dataset."""

import os
import sys
from pathlib import Path

import anndata as ad
import scanpy as sc
import squidpy as sq
from matplotlib.colors import ListedColormap

sys.path.append(
    str(Path(__file__).resolve().parents[3])
)  # goes up 3 levels to AirScape/

from utils.airspace_colors import level_2_listed
from utils.setup_logger import setup_logger


def plot_spatial_distribution(
    adata: sc.AnnData,
    module_dir: Path,
    annotation_key: str | None,
    palette: ListedColormap | None = None,
):
    """Plot spatial distribution of clusters for each ROI.

    Args:
        adata: AnnData with cluster annotations
        module_dir: Directory to save plots
        annotation_key: Annotation column name
        palette: Color palette for the plot
    """
    if annotation_key is None:
        logger.info("No annotation key provided, skipping spatial plots")
        return

    if annotation_key not in adata.obs.columns:
        logger.warning(
            f"Cluster column '{annotation_key}' not found, skipping spatial plots"
        )
        return

    if "ROI" not in adata.obs.columns:
        logger.warning("'ROI' column not found, skipping spatial plots")
        return

    logger.info(f"Plotting spatial distribution: '{annotation_key}'...")

    for roi in adata.obs["ROI"].unique():
        subset = adata[adata.obs["ROI"] == roi]
        sq.pl.spatial_scatter(
            subset,
            library_id="spatial",
            shape=None,
            color=[annotation_key],
            wspace=0.4,
            figsize=(16, 16),
            size=5,
            edgecolor="none",
            palette=palette,
            save=module_dir / f"{annotation_key}_{roi}_spatial.png",
        )

    logger.info(f"Spatial plots saved to {module_dir}")


# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape_analysis/HPC_jobs/general/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk")


# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)

# Set figure directory
folder_name = "spatial_plots"
out_path = f"project_analysis/general/{folder_name}"

out_dir = dir / out_path
os.makedirs(out_dir, exist_ok=True)

# Set scanpy to save figures in the fig/ subdirectory
sc.settings.figdir = out_dir

# Load data
print(f"Loading data from {dir / 'adata_final_object/adata_with_metadata.zarr'}...")
adata = ad.read_zarr(dir / "adata_final_object/adata_with_metadata.zarr")

# Set palette
palette = level_2_listed

# Set cell type annotation levels
annotation_levels = ["level_1", "level_2", "level_3"]

for level in annotation_levels:
    if level in adata.obs:
        adata.obs[level] = adata.obs[level].astype("category")

    # make a new folder for spatial plots for this level
    level_spatial_dir = out_dir / level
    level_spatial_dir.mkdir(exist_ok=True, parents=True)

    # Plot spatial distribution of clusters for this level
    plot_spatial_distribution(
        adata=adata, module_dir=level_spatial_dir, annotation_key=level, palette=palette
    )

logger.info("Dimension reduction module complete.")
