"""UMAP visualizations and cell type composition plots for RejuvenAir project."""

import logging
import os
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from matplotlib import rcParams


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
    celltype_col: str = "level_1",
    groupby_col: str = "condition",
    figsize: tuple = (10, 6),
    palette: str = "tab20b",
    ylabel: str = "Percentage (%)",
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
    legend_bbox : tuple
        Legend position (bbox_to_anchor)

    Returns:
    fig, ax : matplotlib figure and axes objects
    """
    # Make direcotry for cell type
    celltype_col_dir = fig_dir / celltype_col
    os.makedirs(celltype_col_dir, exist_ok=True)

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
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")

    # Customize plot
    ax.set_title(
        f"Cell Type Composition by {groupby_col.replace('_', ' ').title()}",
        fontsize=14,
    )
    ax.set_xlabel(groupby_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 100)

    # Customize legend
    ax.legend(
        title=celltype_col.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(
        celltype_col_dir / f"celltype_composition_{groupby_col}_{celltype_col}.pdf"
    )

    return fig, ax


def plot_celltype_boxplots(
    adata,
    celltype_col: str = "level_1",
    groupby_col: str = "condition",
    donor_col: str = "ROI",
    figsize: tuple = (5, 4),
    palette: str = "tab10",
    ylabel: str = "Percentage (%)",
    stat_test: str = "Mann-Whitney",
    stat_pvalue_format: str = "stars",  # "stars" or "simple"
    fig_dir: Path = Path("."),
    save_fmt: tuple = ("pdf", "png"),
):
    """Plot one boxplot per cell type.

    Each subplot shows the distribution of cell-type percentages across groups.

        - X-axis: groups defined by `groupby_col` (e.g. condition)
        - Y-axis: percentage of that cell type per donor/ROI
        - Points: one swarmplot dot per donor/ROI
        - Statistics: pairwise Mann–Whitney U test with significance brackets

    Args:
        adata (AnnData):
            Annotated data matrix.
        celltype_col (str):
            Column in `adata.obs` containing cell-type labels.
        groupby_col (str):
            Column in `adata.obs` used to define x-axis groups (e.g. 'condition').
        donor_col (str):
            Column in `adata.obs` identifying each donor or ROI.
        figsize (tuple[float, float]):
            Figure size as (width, height) per subplot.
        palette (str | list[str], optional):
            Seaborn or Matplotlib palette name, or list of colors.
            Falls back to `adata.uns[f'{celltype_col}_colors']` if available.
        ylabel (str, optional):
            Label for the y-axis.
        stat_test (str, optional):
            Statistical test passed to statannotations. Defaults to 'Mann-Whitney'.
        stat_pvalue_format (str, optional):
            Format for p-values ('stars' or 'simple'). Defaults to 'stars'.
        fig_dir (str | pathlib.Path, optional):
            Directory where figures will be saved.
        save_fmt (tuple[str, ...], optional):
            File extensions for saving figures (e.g. ('pdf', 'png')).

    Returns:
        dict[str, tuple]:
            Mapping of each cell type to its corresponding (fig, ax) tuple.
    """
    for col in [celltype_col, groupby_col, donor_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    # Make directory for cell type
    celltype_col_dir = fig_dir / celltype_col
    os.makedirs(celltype_col_dir, exist_ok=True)

    # Count cells per (donor, groupby, celltype)
    counts = (
        adata.obs.groupby([donor_col, groupby_col, celltype_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    # Total cells per donor (across all cell types)
    donor_totals = (
        adata.obs.groupby([donor_col, groupby_col], observed=True)
        .size()
        .reset_index(name="total")
    )
    counts = counts.merge(donor_totals, on=[donor_col, groupby_col])
    counts["percentage"] = (counts["count"] / counts["total"]) * 100

    # Colour palette
    cell_types = counts[celltype_col].unique().tolist()
    palette_key = f"{celltype_col}_colors"
    if palette_key in adata.uns:
        cat_order = adata.obs[celltype_col].cat.categories.tolist()
        color_dict = dict(zip(cat_order, adata.uns[palette_key]))
    else:
        colors = sns.color_palette(palette, n_colors=len(cell_types))
        color_dict = dict(zip(cell_types, colors))

    # Group order & pairwise combinations
    groups = counts[groupby_col].unique().tolist()
    pairs = list(combinations(groups, 2))

    # One figure per cell type
    figs = {}
    for ct in cell_types:
        ct_data = counts[counts[celltype_col] == ct].copy()

        # Ensure every group is represented (fill missing donors with 0)
        all_combos = pd.MultiIndex.from_product(
            [counts[donor_col].unique(), groups],
            names=[donor_col, groupby_col],
        )
        ct_data = (
            ct_data.set_index([donor_col, groupby_col])["percentage"]
            .reindex(all_combos, fill_value=0)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=figsize)

        ct_color = color_dict.get(ct, "#888888")
        box_palette = {g: ct_color for g in groups}

        # Boxplot
        sns.boxplot(
            data=ct_data,
            x=groupby_col,
            y="percentage",
            order=groups,
            palette=box_palette,
            width=0.5,
            fliersize=0,  # hide default outlier markers; swarm handles points
            linewidth=1.2,
            boxprops=dict(alpha=0.55),
            ax=ax,
        )

        # Swarmplot (one dot per donor)
        sns.swarmplot(
            data=ct_data,
            x=groupby_col,
            y="percentage",
            order=groups,
            color="black",
            size=5,
            alpha=0.85,
            ax=ax,
        )

        # # ── Statistical annotations ──────────────────────────────────────────
        # if len(pairs) > 0:
        #     try:
        #         annotator = Annotator(
        #             ax,
        #             pairs,
        #             data=ct_data,
        #             x=groupby_col,
        #             y="percentage",
        #             order=groups,
        #         )
        #         annotator.configure(
        #             test=stat_test,
        #             text_format=stat_pvalue_format,
        #             loc="outside",
        #             verbose=0,
        #             comparisons_correction=None,   # no multiple-testing correction
        #             hide_non_significant=True,     # only show significant brackets
        #         )
        #         annotator.apply_and_annotate()
        #     except Exception as e:
        #         print(f"  [stats warning for '{ct}']: {e}")

        #  Aesthetics
        ax.set_title(ct, fontsize=13, fontweight="bold")
        ax.set_xlabel(groupby_col.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)
        plt.tight_layout()

        # Save
        safe_ct = ct.replace("/", "-").replace(" ", "_")
        for fmt in save_fmt:
            out_path = (
                celltype_col_dir
                / f"boxplot_{groupby_col}_{celltype_col}_{safe_ct}.{fmt}"
            )
            fig.savefig(out_path, bbox_inches="tight", dpi=150)

        figs[ct] = (fig, ax)
        plt.close(fig)

    print(f"Saved {len(figs)} figures to '{celltype_col_dir}'")
    return figs


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/rejuvenair/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="umaps")

# Figure parameters
FIGSIZE = (6, 5)
rcParams["figure.figsize"] = FIGSIZE

# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)

# Set figure directory
folder_name = "umaps"
fig_path = f"project_analysis/RejuvenAir_MICAIII/{folder_name}"

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

# Plot UMAP colored

# Plot UMAP
umap_list = [
    "timepoint",
    "treatment_arm",
    "time_and_treatment_arm",
    "batch",
    "gender",
    "sample_ID",
    "level_1",
    "level_2",
]

umap_list = [col for col in umap_list if col in adata.obs.columns]

for col in umap_list:
    sc.pl.umap(
        adata,
        color=col,
        frameon=False,
        palette="tab20b",
        title=f"UMAP colored by {col}",
        save=f"UMAP_{col}.png",
    )


FIGSIZE = (10, 8)
# Composition stacked bar charts
comp_list = ["level_1", "level_2", "level_3"]
for col in comp_list:
    fig, ax = plot_celltype_composition(
        adata,
        celltype_col=col,
        groupby_col="time_and_treatment_arm",
    )


# # Boxplots

# figs = plot_celltype_boxplots(
#     adata,
#     celltype_col="level_1",
#     groupby_col="condition",
#     donor_col="ROI",
#     fig_dir=fig_dir,
#     figsize=(4, 4),
#     stat_pvalue_format="stars",
# )
