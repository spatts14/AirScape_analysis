"""Plot PCA and spatial distribution of clusters for the full dataset."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib.colors import ListedColormap

from utils.seed_everything import seed_everything
from utils.setup_logger import setup_logger


def plot_dimensionality_reduction(
    adata: sc.AnnData,
    module_dir: Path,
    norm_approach: str,
    n_neighbors: int,
    leiden_keys: str | Sequence[str] = "leiden_best",
    cmap: str = "crest",
):
    """Create and save dimensionality reduction plots.

    Args:
        adata: AnnData with computed UMAP and clusters
        module_dir: Directory to save figures
        norm_approach: Normalization approach label
        n_neighbors: Number of neighbors used
        leiden_keys: Leiden clustering column name
        cmap: Colormap for continuous variables
    """
    if cmap is None:
        cmap = sns.color_palette("crest", as_cmap=True)

    logger.info("Plotting UMAPs...")

    for key in leiden_keys:
        if key not in adata.obs.columns:
            logger.warning(f"Cluster column '{key}' not found, skipping")
            continue

        logger.info(f"Plotting UMAP for '{key}'")

        color_list = [
            "total_counts",
            "n_genes_by_counts",
            key,
        ]

        sc.pl.umap(
            adata,
            color=color_list,
            ncols=3,
            cmap=cmap,
            wspace=0.4,
            show=False,
            frameon=False,
            save=f"_{key}_{norm_approach}_n{n_neighbors}.pdf",
        )

    logger.info(f"Plots saved to {module_dir}")


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


def get_palette(n):
    """Generate a color palette with n distinct colors.

    Args:
        n: Number of distinct colors needed.

    Returns:
        A ListedColormap with n distinct colors.
    """
    colors = [plt.cm.tab20(i) for i in np.linspace(0, 1, 20)] + [
        plt.cm.tab20b(i) for i in np.linspace(0, 1, 20)
    ]
    return ListedColormap(colors[:n])


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/general/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk")


# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)
data_dir = dir / "project_analysis/general/pseudobulk_data/"

# Set figure directory
folder_name = "pca"
out_path = f"project_analysis/general/{folder_name}"

out_dir = dir / out_path
os.makedirs(out_dir, exist_ok=True)


# Create fig subdirectory for scanpy figures
fig_dir = out_dir
fig_dir.mkdir(exist_ok=True, parents=True)

spatial_plots_dir = fig_dir / "spatial_plots"
spatial_plots_dir.mkdir(exist_ok=True, parents=True)

# Set scanpy to save figures in the fig/ subdirectory
sc.settings.figdir = fig_dir
cmap = sns.color_palette("crest", as_cmap=True)
palette = sns.color_palette("Set2", 12)

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "annotate/adata_level_2_level_3.h5ad")  # full dataset

# Ensure PCA is computed (only compute if not already present)
if "X_pca" not in adata.obsm:
    logger.info(f"Computing PCA (n_comps={60})...")
    sc.pp.pca(adata, n_comps=60, svd_solver="arpack", use_highly_variable=True)
else:
    logger.info("PCA already computed, skipping...")


# Visualize PC1 vs PC2 colored by total_counts and n_genes_by_counts
logger.info("Plotting PCA scatter plots...")


# Plot PCA with observation fields if available
annotation_levels = ["level_1", "level_2", "level_3"]
for level in annotation_levels:
    logger.info("Plotting PCA with observation fields...")
    if level in adata.obs.columns:
        sc.pl.pca(
            adata,
            color=level,
            palette=palette,
            dimensions=[(0, 1)],
            ncols=3,
            size=2,
            alpha=0.5,
            show=False,
            save=f"_{level}_PC1_PC2.png",
        )
    if level in adata.obs.columns:
        sc.pl.pca(
            adata,
            color=level,
            palette=palette,
            dimensions=[(2, 3)],
            ncols=3,
            size=2,
            alpha=0.5,
            show=False,
            save=f"_{level}_PC3_PC4.png",
        )
else:
    logger.info("Skipping PCA observation fields plot (obs_vis_list not configured)")

for level in annotation_levels:
    if level in adata.obs:
        adata.obs[level] = adata.obs[level].astype("category")

    # make a new folder for spatial plots for this level
    level_spatial_dir = spatial_plots_dir / level
    level_spatial_dir.mkdir(exist_ok=True, parents=True)

    # Make color palette for this level
    palette = get_palette(len(adata.obs[level].cat.categories))

    # Plot spatial distribution of clusters for this level
    plot_spatial_distribution(
        adata=adata, module_dir=level_spatial_dir, annotation_key=level, palette=palette
    )

logger.info("Dimension reduction module complete.")
