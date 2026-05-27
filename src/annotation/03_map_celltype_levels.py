"""Map cell type annotations to clusters for specified levels."""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import seaborn as sns

from utils.seed_everything import seed_everything


def load_annotation_dict_from_env(var_name: str) -> dict[str, str]:
    """Load a JSON dictionary from an environment variable."""
    raw_value = os.getenv(var_name)
    if not raw_value:
        raise ValueError(f"{var_name} environment variable must be set")

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'{var_name} must be valid JSON, for example: \'{{"0": "A"}}\''
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{var_name} must decode to a JSON object")

    return {str(key): str(value) for key, value in parsed.items()}


def map_clusters_to_annotations(
    adata,
    subset: str,
    chosen_resolution: float,
    annotation_dict: dict,
    annotation_level: str,
    logger: logging.Logger | None = None,
) -> None:
    """Map Leiden clusters at a chosen resolution to cell type annotations.

    This function maps cluster labels from a specified Leiden resolution
    to user-defined annotation labels and stores the result in a new
    categorical column in ``adata.obs``.

    Args:
        adata: AnnData object containing clustering results.
        subset: Prefix used for Leiden clustering keys (e.g., "airway_epithelium").
        chosen_resolution: Resolution value corresponding to the
            clustering column (e.g., 0.5).
        annotation_dict: Dictionary mapping cluster labels to
            annotation names (e.g., {"0": "Naive", "1": "Memory"}).
        annotation_level: Name of the new column in ``adata.obs``
            where annotations will be stored.
        logger: Optional logger for reporting progress and warnings.

    Returns:
        None. The function modifies ``adata.obs`` in-place.

    Raises:
        None explicitly. Logs warnings if the cluster key is missing,
        the annotation dictionary is empty, or some clusters are not mapped.
    """
    # Resolve the cluster column name based on subset and chosen resolution
    rename_subset_key = chosen_resolution_name

    # Check cluster column exists
    if rename_subset_key not in adata.obs:
        if logger:
            logger.warning(
                "%s not found in adata.obs. Cannot map clusters.",
                rename_subset_key,
            )
        return

    # Check mapping provided
    if not annotation_dict:
        if logger:
            logger.warning("annotation_dict is empty or None. Cannot map clusters.")
        return

    if logger:
        logger.info("Mapping clusters to cell type annotations.")

    mapped = adata.obs[rename_subset_key].map(annotation_dict)

    # Check for unmapped clusters
    if mapped.isna().any():
        missing = adata.obs.loc[mapped.isna(), rename_subset_key].unique().tolist()
        if logger:
            logger.warning("Some clusters were not mapped: %s", missing)

    # Store as categorical
    adata.obs[annotation_level] = mapped.astype("category")

    if logger:
        distribution = adata.obs[annotation_level].value_counts().to_dict()
        logger.info(
            "Annotation mapping completed. Cell type distribution: %s",
            distribution,
        )


def viz_annotation_level(
    adata,
    annotation_level: str,
    fig_dir: Path,
    file_dir: Path,
    subset: str,
    cmap,
    logger,
    condition_key: str = "condition",
    roi_key: str = "ROI",
    cell_id_key: str = "cell_id",
    n_top_genes: int = 5,
) -> None:
    """Visualize a given annotation level in AnnData.

    Args:
        adata: AnnData
            Annotated data matrix.
        annotation_level: str
            Column name in adata.obs containing annotations.
        fig_dir: Path
            Directory to save figures.
        file_dir: Path
            Directory to save output CSV files.
        subset: str
            Name used as prefix for output files.
        cmap: matplotlib colormap
            Colormap for dotplot.
        logger: logging.Logger
            Logger instance.
        condition_key: str, optional
        Column name for condition (default: "condition").
        roi_key : str, optional
            Column name for ROI (default: "ROI").
        cell_id_key : str, optional
            Column name for cell IDs (default: "cell_id").
        n_top_genes : int, optional
            Number of marker genes to plot (default: 5).
    """
    if annotation_level not in adata.obs:
        logger.warning(
            f"Annotation column '{annotation_level}' was not found. "
            "Skipping annotation-dependent outputs."
        )
        return

    logger.info(f"Annotation column '{annotation_level}' found. Processing...")

    # Set figure directory
    sc.settings.figdir = fig_dir

    # UMAP visualization
    sc.pl.umap(
        adata,
        color=annotation_level,
        legend_loc="right margin",
        legend_fontsize=14,
        frameon=False,
        ncols=2,
        wspace=0.4,
        save=f"_{annotation_level}_umap.pdf",
    )

    # Rank genes
    rank_key = f"rank_genes_{annotation_level}"
    sc.tl.rank_genes_groups(
        adata,
        groupby=annotation_level,
        method="wilcoxon",
        key_added=rank_key,
    )

    # Dendrogram
    dendro_key = f"dendrogram_{annotation_level}"
    sc.tl.dendrogram(
        adata,
        groupby=annotation_level,
        key_added=dendro_key,
    )

    # Dotplot
    sc.pl.rank_genes_groups_dotplot(
        adata,
        groupby=annotation_level,
        dendrogram=dendro_key,
        standard_scale="var",
        n_genes=n_top_genes,
        key=rank_key,
        cmap=cmap,
        save=f"_{annotation_level}_dotplot.pdf",
    )

    # Cell type counts
    counts = adata.obs[annotation_level].value_counts().to_frame(name="count")
    counts.to_csv(file_dir / f"_{annotation_level}_{subset}_celltype_counts.csv")

    # Counts per condition and ROI
    if condition_key in adata.obs and roi_key in adata.obs:
        counts_condition = pd.crosstab(
            adata.obs[annotation_level],
            [adata.obs[condition_key], adata.obs[roi_key]],
        )
        counts_condition.to_csv(
            file_dir
            / f"_{annotation_level}_{subset}_celltype_counts_per_condition_ROI.csv"
        )
    else:
        logger.warning("Condition or ROI columns missing; skipping crosstab.")

    # Export annotations
    if cell_id_key in adata.obs:
        df = adata.obs[[annotation_level, cell_id_key]].copy()
        df.columns = ["cell_annotation", "cell_id"]
        df.to_csv(
            file_dir / f"_{annotation_level}_{subset}_annotations.csv", index=False
        )
    else:
        logger.warning(f"'{cell_id_key}' not found in adata.obs; skipping export.")

    logger.info(f"Finished processing annotation level '{annotation_level}'.")


# Set random seed for reproducibility
seed_everything(19960915)

# Read directory from environment variable
celltype_subset_dir = os.getenv("CELLTYPE_SUBSET_DIR")
if not celltype_subset_dir:
    raise ValueError("CELLTYPE_SUBSET_DIR environment variable must be set")
module_dir = Path(celltype_subset_dir)

# Set variables
h5ad_file = os.getenv("H5AD_FILE")
subset = os.getenv("SUBSET")
resolution = os.getenv("RESOLUTION")
level_2_annotation_dict = load_annotation_dict_from_env("LEVEL2_ANNOTATION_DICT")
level_3_annotation_dict = load_annotation_dict_from_env("LEVEL3_ANNOTATION_DICT")


# chosen_resolution_name = f"Immune_cells_{resolution}"
chosen_resolution_name = f"{subset}_{resolution}"

# Define annotation column names
annotation_level = "level_1"
annotation_level_1 = "level_1"
annotation_level_2 = "level_2"
annotation_level_3 = "level_3"


# Set up directories
subset_dir = module_dir / subset
subset_dir.mkdir(parents=True, exist_ok=True)

# Save figures
fig_dir = subset_dir / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = fig_dir

# Save files
file_dir = subset_dir / "files"
file_dir.mkdir(parents=True, exist_ok=True)


# Set up logging
log_file = subset_dir / f"celltype_{annotation_level}_{subset}.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Set colors
cmap_blue = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
color_palette_level_1 = sns.color_palette("hls", 12)

# Load data
logger.info(f"Loading data from {module_dir / f'{h5ad_file}'}")
adata = sc.read_h5ad(module_dir / f"{h5ad_file}")
logger.info(f"Data loaded successfully. Shape: {adata.shape}")

# Check available columns and log cell type information
logger.info(f"Available columns in adata.obs: {list(adata.obs.columns)}")

# UMAP plot colored by manual annotation (i.e coarse clustering)
sc.pl.umap(
    adata,
    color=annotation_level,
    legend_loc="right margin",
    legend_fontsize=14,
    frameon=False,
    ncols=2,
    wspace=0.4,
    save=f"_{annotation_level_1}.pdf",
)

logger.info(f"Starting Level 1 cell type annotation for subset: {subset}")
logger.info(f"Output directory: {subset_dir}")

# # Map level 2 annotations
map_clusters_to_annotations(
    adata,
    subset=subset,
    chosen_resolution=chosen_resolution_name,
    annotation_dict=level_2_annotation_dict,
    annotation_level=annotation_level_2,
    logger=logger,
)

# Map level 3 annotations
map_clusters_to_annotations(
    adata,
    subset=subset,
    chosen_resolution=chosen_resolution_name,
    annotation_dict=level_3_annotation_dict,
    annotation_level=annotation_level_3,
    logger=logger,
)

# Visualize annotation levels
viz_annotation_level(
    adata=adata,
    annotation_level=annotation_level_2,
    fig_dir=fig_dir,
    file_dir=file_dir,
    subset=subset,
    cmap=cmap_blue,
    logger=logger,
)

viz_annotation_level(
    adata=adata,
    annotation_level=annotation_level_3,
    fig_dir=fig_dir,
    file_dir=file_dir,
    subset=subset,
    cmap=cmap_blue,
    logger=logger,
)

# Save the annotated data
output_file = module_dir / f"adata_subset_{subset}.h5ad"
logger.info(f"Saving annotated data to {output_file}")
adata.write_h5ad(output_file)
logger.info("Main analysis data saved successfully")
logger.info(f"Final data shape: {adata.shape}")

logger.info(f"Analysis completed successfully for {subset}")
print("Done!")
