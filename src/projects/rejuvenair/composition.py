"""Generate composition of celltype plots."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import torch


# Functions
def seed_everything(seed: int):
    """Set random seed on every random module for reproducibility.

    Args:
        seed: The seed value to set for random number generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        pass


def setup_logger(log_dir: Path, log_name: str) -> logging.Logger:
    """Set up a logger that writes to both console and a timestamped file.

    Args:
        log_dir: Directory where the log file will be saved.
        log_name: Base name for the log file.

    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if function is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


def plot_celltype_composition(
    adata,
    celltype_col: str = "level_2",
    groupby_col: str = "condition",
    figsize: tuple = (6, 6),
    palette: str = "tab20b",
    ylabel: str = "Percentage (%)",
    celltype_subset: str = None,
    file_name: str = "celltype_composition.pdf",
    legend_bbox: tuple = (1.02, 1),
):
    """Plot stacked bar chart showing cell type composition per condition.

    Args:
    adata : AnnData
        Annotated data object with cell type and condition info in .obs
    celltype_col : str
        Column name in adata.obs containing cell type annotations
    groupby_col : str
        Column name in adata.obs to group by (e.g., 'condition', 'sample')
    figsize : tuple
        Figure size (width, height)
    palette : str
        Seaborn/matplotlib color palette name
    title : str
        Plot title (default: auto-generated)
    ylabel : str
        Y-axis label
    celltype_subset : Subset of cell type for calculating percentage composition.
    (e.g Airway epithelial cells)
    If None, percentage is calculated based on all cell types.
    file_name : str
        Name of the file to save the plot
    legend_bbox : tuple
        Legend position (bbox_to_anchor)

    Returns:
    fig, ax : matplotlib figure and axes objects
    """
    # Validate columns exist
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_col}' not found in adata.obs")
    if groupby_col not in adata.obs.columns:
        raise ValueError(f"Column '{groupby_col}' not found in adata.obs")

    # Calculate counts and percentages
    counts = (
        adata.obs.groupby([groupby_col, celltype_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(groupby_col)["count"].transform("sum")
    counts["percentage"] = (counts["count"] / totals) * 100

    # Pivot for stacking
    pivot_df = counts.pivot(
        index=groupby_col, columns=celltype_col, values="percentage"
    ).fillna(0)

    # Get unique cell types and colors
    cell_types = pivot_df.columns.tolist()

    # Check if color palette exists in adata.uns, otherwise create new one
    palette_key = f"{celltype_col}_colors"
    if palette_key in adata.uns:
        print(f"Using existing color palette from adata.uns['{palette_key}']")
        existing_colors = adata.uns[palette_key]
        # Create mapping from cell types to colors
        cell_type_categories = adata.obs[celltype_col].cat.categories.tolist()
        color_dict = dict(zip(cell_type_categories, existing_colors))
        # Filter to only the cell types present in the plot
        color_dict = {ct: color_dict[ct] for ct in cell_types if ct in color_dict}
    else:
        print(f"No existing palette found, creating new one with '{palette}'")
        colors = sns.color_palette(palette, n_colors=len(cell_types))
        color_dict = dict(zip(cell_types, colors))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked bars using seaborn barplot for each layer
    x_positions = np.arange(len(pivot_df.index))
    bar_width = 0.7
    bottom = np.zeros(len(pivot_df.index))

    for cell_type in cell_types:
        values = pivot_df[cell_type].values
        sns.barplot(
            x=x_positions,
            y=values,
            color=color_dict[cell_type],
            label=cell_type,
            bottom=bottom,
            ax=ax,
            width=bar_width,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    # Set x-tick labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index, rotation=90, ha="right")

    # Customize plot
    ax.set_title(
        f"Cell Type Composition by {groupby_col.replace('_', ' ').title()}",
        fontsize=10,
    )
    ax.set_xlabel(groupby_col.replace("_", " ").title(), fontsize=14)
    ylabel = f"{ylabel} of {celltype_subset.replace('_', ' ').title()}"
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0, 100)

    # Customize legend
    ax.legend(
        title=celltype_col.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    if file_name:
        plt.savefig(fig_dir / file_name, bbox_inches="tight")
    else:
        plt.savefig(
            fig_dir / f"celltype_composition_{groupby_col}_{celltype_col}.pdf",
            bbox_inches="tight",
        )

    return fig, ax


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/rejuvenair/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="composition")

# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)

# Set figure directory
fig_path = "project_analysis/RejuvenAir_MICAIII/composition"

fig_dir = dir / fig_path
os.makedirs(fig_dir, exist_ok=True)

# set fig dir for plots to save to
sc.settings.figdir = fig_dir

# Set colors
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
color_palette_level_1 = sns.color_palette("hls", 12)

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "subset_adata/adata_subset_COPD_MICA_clean.h5ad")
logger.info(adata)

logger.info("Data loaded successfully.")

# Subset to specific arm only for plotting
logger.info("Subsetting data by treatment arm...")
adata_treatment = adata[adata.obs["treatment_arm"].isin(["Treatment"])].copy()
adata_sham = adata[adata.obs["treatment_arm"].isin(["Sham"])].copy()

# Plot
logger.info("Plotting cell type composition...")

level_1_list = adata.obs["level_1_corrected"].unique().tolist()
logger.info(f"Unique level_1_corrected categories: {level_1_list}")

for subset in level_1_list:
    subset_adata = adata[adata.obs["level_1_corrected"] == subset].copy()
    for col in ["ROI", "time_and_treatment_arm"]:
        if col not in subset_adata.obs.columns:
            logger.warning(
                f"Warning: Column '{col}' not found in adata.obs for subset '{subset}'"
            )
        else:
            logger.info(
                f"Plot composition for subset '{subset}' - {subset_adata.n_obs} cells"
            )
            fig, ax = plot_celltype_composition(
                subset_adata,
                figsize=(8, 6),
                celltype_col="level_2",
                groupby_col=col,
                celltype_subset=subset,
                file_name=f"celltype_composition_{subset}_{col}.pdf",
            )


# Plot treatment arm composition by condition, ROI, and timepoint for each level_1 subset
for subset in level_1_list:
    subset_adata = adata_treatment[
        adata_treatment.obs["level_1_corrected"] == subset
    ].copy()
    for col in ["ROI", "timepoint"]:
        if col not in subset_adata.obs.columns:
            logger.warning(
                f"Warning: Column '{col}' not found in adata.obs for subset '{subset}'"
            )
        else:
            logger.info(
                f"Plot composition for subset '{subset}' - {subset_adata.n_obs} cells"
            )
            fig, ax = plot_celltype_composition(
                subset_adata,
                figsize=(8, 6),
                celltype_col="level_2",
                groupby_col=col,
                celltype_subset=subset,
                file_name=f"celltype_composition_{subset}_{col}_treatment.pdf",
            )

# Plot sham arm composition by condition, ROI, and timepoint for each level_1 subset
for subset in level_1_list:
    subset_adata = adata_sham[adata_sham.obs["level_1_corrected"] == subset].copy()
    for col in ["ROI", "timepoint"]:
        if col not in subset_adata.obs.columns:
            logger.warning(
                f"Warning: Column '{col}' not found in adata.obs for subset '{subset}'"
            )
        else:
            logger.info(
                f"Plot composition for subset '{subset}' - {subset_adata.n_obs} cells"
            )
            fig, ax = plot_celltype_composition(
                subset_adata,
                figsize=(8, 6),
                celltype_col="level_2",
                groupby_col=col,
                celltype_subset=subset,
                file_name=f"celltype_composition_{subset}_{col}_sham.pdf",
            )

logger.info("All plots generated successfully.")
