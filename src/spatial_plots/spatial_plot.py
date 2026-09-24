"""Spatial plots for the AirScape analysis."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import squidpy as sq
from matplotlib.colors import ListedColormap, Normalize

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.airspace_colors import level_2_listed
from utils.setup_logger import setup_logger


def _gene_values(adata: sc.AnnData, gene: str) -> np.ndarray:
    """Return a 1D dense array of expression values for one gene from .X."""
    x = adata[:, gene].X
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).ravel()


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


def plot_spatial_gene_expression(
    adata: sc.AnnData,
    module_dir: Path,
    gene_list: list[str],
    cmap: str | ListedColormap | None = "mako",
):
    """Plot spatial expression of each gene in gene_list, for every ROI.

    Creates one subfolder per gene inside module_dir, and saves one spatial
    plot per ROI into that gene's subfolder. For each gene, the color scale
    is shared across all ROIs: the min and max are computed per ROI, and the
    lowest min and highest max are used for every ROI of that gene.

    Args:
        adata: AnnData with spatial coordinates in .obsm["spatial"]
        module_dir: Base directory to save plots; a subfolder is created
            per gene inside this directory
        gene_list: List of gene names to plot spatially
        cmap: Colormap for expression values (a sequential colormap,
            since gene expression is continuous)
    """
    if "ROI" not in adata.obs.columns:
        logger.warning("'ROI' column not found, skipping spatial gene plots")
        return

    genes_found = [g for g in gene_list if g in adata.var_names]
    genes_missing = [g for g in gene_list if g not in adata.var_names]
    if genes_missing:
        logger.warning(f"Genes not found in adata.var_names, skipped: {genes_missing}")

    if not genes_found:
        logger.warning("None of the requested genes were found, skipping.")
        return

    rois = adata.obs["ROI"].unique()

    for gene in genes_found:
        gene_dir = module_dir / gene
        gene_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Plotting spatial expression for gene '{gene}'...")

        # Per-ROI min/max for this gene
        roi_ranges = {}
        for roi in rois:
            values = _gene_values(adata[adata.obs["ROI"] == roi], gene)
            roi_ranges[roi] = (float(values.min()), float(values.max()))

        # Shared gene-specific scale: widest range across ROIs
        vmin = min(r[0] for r in roi_ranges.values())
        vmax = max(r[1] for r in roi_ranges.values())
        if vmax == vmin:
            vmax = vmin + 1e-6  # avoid a zero-width color scale
        norm = Normalize(vmin=vmin, vmax=vmax)

        logger.info(f"'{gene}' shared scale: vmin={vmin:.3g}, vmax={vmax:.3g}")

        for roi in rois:
            subset = adata[adata.obs["ROI"] == roi]
            safe_roi_name = str(roi).replace(" ", "_").replace("/", "-")

            sq.pl.spatial_scatter(
                subset,
                library_id="spatial",
                shape=None,
                color=[gene],
                use_raw=False,  # read the same values (.X) the scale was computed from
                norm=norm,
                wspace=0.4,
                figsize=(16, 16),
                size=5,
                edgecolor="none",
                cmap=cmap,
                save=gene_dir / f"{gene}_{safe_roi_name}_spatial.png",
            )

        logger.info(f"Saved spatial plots for '{gene}' to {gene_dir}")


def plot_spatial_score(
    adata: sc.AnnData,
    module_dir: Path,
    score_name: str,
    cmap: str | ListedColormap | None = "mako",
):
    """Plot spatial distribution of a continuous score for every ROI.

    Args:
        adata: AnnData with spatial coordinates in .obsm["spatial"] and the
            score stored as a column in .obs
        module_dir: Directory to save plots; created if it doesn't exist
        score_name: Name of the .obs column holding the continuous score
        cmap: Colormap for the score (continuous, so a sequential colormap
            like "mako" is used, not a categorical palette)
    """
    module_dir.mkdir(exist_ok=True, parents=True)

    if score_name not in adata.obs.columns:
        logger.warning(f"Score column '{score_name}' not found in adata.obs, skipping.")
        return

    if "ROI" not in adata.obs.columns:
        logger.warning("'ROI' column not found, skipping spatial score plot")
        return

    logger.info(f"Plotting spatial distribution for score '{score_name}'...")

    for roi in adata.obs["ROI"].unique():
        subset = adata[adata.obs["ROI"] == roi]
        safe_roi_name = str(roi).replace(" ", "_").replace("/", "-")

        sq.pl.spatial_scatter(
            subset,
            library_id="spatial",
            shape=None,
            color=[score_name],
            wspace=0.4,
            figsize=(16, 16),
            size=5,
            edgecolor="none",
            cmap=cmap,
            save=module_dir / f"{score_name}_{safe_roi_name}_spatial.png",
        )

    logger.info(f"Spatial score plots saved to {module_dir}")


# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape_analysis/HPC_jobs/general/"
logs_dir = Path(wd) / "logs" / "spatial_plots"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="spatial_plots")

# Set directories
path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)
dir = path / "output/AIRSCAPE/"

fig_dir = dir / "spatial_plots"
fig_dir.mkdir(parents=True, exist_ok=True)
gene_spatial_dir = fig_dir / "gene_expression"
gene_spatial_dir.mkdir(exist_ok=True, parents=True)


# Configure scanpy to save figures in our custom directory
sc.settings.figdir = fig_dir

# Load data
print("Loading data from 'adata_final_object/adata_with_metadata.zarr'...")
adata = ad.read_zarr(dir / "adata_final_object/adata_with_metadata.zarr")

# Set palette
palette = level_2_listed
cmap = sns.color_palette("mako", as_cmap=True)

# Set cell type annotation levels
annotation_levels = ["level_1", "level_2"]
gene_list = [
    # TGF-β activation
    "TGFB1",
    "ITGB6",
    "THBS1",
    "CCN2",
    "TGFBI",
    # Fibrillar and other collagens
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL5A1",
    "COL6A1",
    "COL6A2",
    "COL6A3",
    "COL14A1",
    # Glycoproteins and matricellular proteins
    "FN1",
    "POSTN",
    "CTHRC1",
    "SPARC",
    "TNC",
    "COMP",
    "FBN1",
    "ELN",
    # Proteoglycans
    "LUM",
    "VCAN",
    "BGN",
    "ASPN",
    "FMOD",
    # Myofibroblast marker
    "ACTA2",
    # Crosslinking
    "LOX",
    "LOXL1",
    "LOXL2",
    "PLOD2",
    # Turnover and biomarkers
    "MMP1",
    "MMP2",
    "MMP7",
    "MMP14",
    "TIMP1",
    "SERPINE1",
    "SPP1",
    # Developmental pathways (representative genes)
    "CTNNB1",
    "WNT5A",
    "AXIN2",  # Wnt/β-catenin
    "SHH",
    "GLI1",  # Hedgehog
    "NOTCH1",
    "JAG1",
    "HES1",  # Notch
]

score_name = "fibroblast_remodeling_score"
gene_score_list = gene_list

# for level in annotation_levels:
#     if level in adata.obs:
#         adata.obs[level] = adata.obs[level].astype("category")

#     # make a new folder for spatial plots for this level
#     level_spatial_dir = fig_dir / level
#     level_spatial_dir.mkdir(exist_ok=True, parents=True)

#     # Plot spatial distribution of clusters for this level
#     plot_spatial_distribution(
#         adata=adata, module_dir=level_spatial_dir, annotation_key=level, palette=palette
#     )

# Gene-level spatial expression plots
gene_present = [g for g in gene_list if g in adata.var_names]

# spatial gene expression plots
plot_spatial_gene_expression(
    adata=adata,
    module_dir=gene_spatial_dir,
    gene_list=gene_present,
    cmap=cmap,
)

# Dotplot of genes in list
sc.pl.dotplot(
    adata,
    var_names=gene_present,
    groupby=["level_2"],
    standard_scale="var",
    cmap=cmap,
    save="_level_2.pdf",
)

# Plot dotplot of genes in list, grouped by level_2 and condition
mask = adata.obs["level_2"].notna() & adata.obs["condition"].notna()
n_dropped = (~mask).sum()
if n_dropped > 0:
    logger.warning(
        f"Dropping {n_dropped} cells with missing 'level_2' or 'condition' "
        "before combined dotplot."
    )
subset_for_combined = adata[mask]

sc.pl.dotplot(
    subset_for_combined,
    var_names=gene_present,
    groupby=["level_2", "condition"],
    standard_scale="var",
    cmap=cmap,
    save="_level_2_condition.pdf",
)


# Gene score spatial expression plots
gene_score_present = [g for g in gene_score_list if g in adata.var_names]

sc.tl.score_genes(
    adata,
    gene_list=gene_score_present,
    score_name=score_name,
)

plot_spatial_score(
    adata=adata,
    module_dir=fig_dir,
    score_name=score_name,
    cmap=cmap,
)

logger.info("Spatial plot module complete.")
