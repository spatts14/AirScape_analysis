"""Generate composition of celltype plots."""

import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.seed_everything import seed_everything


def plot_celltype_composition(
    adata,
    celltype_col: str = "level_2",
    groupby_col: str = "condition",
    figsize: tuple = (10, 6),
    palette: str = "tab20",
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
    ylabel : str
        Y-axis label
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
        cell_type_categories = adata.obs[celltype_col].cat.categories.tolist()
        color_dict = dict(zip(cell_type_categories, existing_colors))
        color_dict = {ct: color_dict[ct] for ct in cell_types if ct in color_dict}
    else:
        print(f"No existing palette found, creating new one with '{palette}'")
        colors = sns.color_palette(palette, n_colors=len(cell_types))
        color_dict = dict(zip(cell_types, colors))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked bars
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

    ax.legend(
        title=celltype_col.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(fig_dir / f"celltype_composition_{groupby_col}_{celltype_col}.pdf")

    return fig, ax


def plot_level2_within_level1(
    adata,
    level1_col: str = "level_1",
    level2_col: str = "level_2",
    groupby_col: str = "diagnosis",
    figsize: tuple = (10, 6),
    palette: str = "tab20",
    ylabel: str = "Percentage (%)",
):
    """Plot showing level 2 cell type composition within each level 1 group.

    For example, for level 1 = "Airway Epithelium", the plot shows what
    percentage of airway cells are goblet, ciliated, basal, etc., broken down
    by diagnosis.

    Args:
        adata : AnnData
            Annotated data object with obs columns for level 1, level 2, and groupby.
        level1_col : str
            Column in adata.obs with coarse (level 1) cell type annotations.
        level2_col : str
            Column in adata.obs with fine (level 2) cell type annotations.
        groupby_col : str
            Column in adata.obs to group bars by (e.g. 'diagnosis', 'condition', 'ROI').
        figsize : tuple
            Figure size (width, height) for each individual plot.
        palette : str
            Seaborn/matplotlib color palette name, used only if no existing palette
            is found in adata.uns.
        ylabel : str
            Y-axis label.

    Returns:
        figs : dict
            Dictionary mapping level 1 group name -> (fig, ax) tuple.
    """
    # Validate columns
    for col in [level1_col, level2_col, groupby_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    # Build a color dict for level 2 cell types
    palette_key = f"{level2_col}_colors"
    all_level2 = adata.obs[level2_col].cat.categories.tolist()

    if palette_key in adata.uns:
        print(f"Using existing color palette from adata.uns['{palette_key}']")
        existing_colors = adata.uns[palette_key]
        color_dict = dict(zip(all_level2, existing_colors))
    else:
        print(f"No existing palette found, creating new one with '{palette}'")
        colors = sns.color_palette(palette, n_colors=len(all_level2))
        color_dict = dict(zip(all_level2, colors))

    level1_groups = adata.obs[level1_col].unique()
    figs = {}

    for l1_group in sorted(level1_groups):
        # Subset to cells belonging to this level 1 group
        mask = adata.obs[level1_col] == l1_group
        subset = adata.obs[mask]

        if subset.shape[0] == 0:
            print(f"Skipping '{l1_group}': no cells found.")
            continue

        # Count level 2 subtypes within each groupby value
        counts = (
            subset.groupby([groupby_col, level2_col], observed=True)
            .size()
            .reset_index(name="count")
        )

        # Drop level 2 subtypes with zero total counts in this subset
        level2_present = counts.loc[counts["count"] > 0, level2_col].unique().tolist()
        counts = counts[counts[level2_col].isin(level2_present)]

        # Calculate percentage of each level 2 subtype within each groupby value
        totals = counts.groupby(groupby_col)["count"].transform("sum")
        counts["percentage"] = (counts["count"] / totals) * 100

        # Pivot for stacking
        pivot_df = counts.pivot(
            index=groupby_col, columns=level2_col, values="percentage"
        ).fillna(0)

        cell_types = pivot_df.columns.tolist()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        x_positions = np.arange(len(pivot_df.index))
        bar_width = 0.7
        bottom = np.zeros(len(pivot_df.index))

        for cell_type in cell_types:
            values = pivot_df[cell_type].values
            sns.barplot(
                x=x_positions,
                y=values,
                color=color_dict.get(cell_type, "#aaaaaa"),
                label=cell_type,
                bottom=bottom,
                ax=ax,
                width=bar_width,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += values

        # X-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")

        # Titles and labels
        l1_title = l1_group.replace("_", " ").title()
        groupby_title = groupby_col.replace("_", " ").title()
        ax.set_title(
            f"{l1_title}: Level 2 Composition by {groupby_title}",
            fontsize=14,
        )
        ax.set_xlabel(groupby_title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 100)

        ax.legend(
            title=level2_col.replace("_", " ").title(),
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

        plt.tight_layout()

        # Save with a safe filename (replace spaces/slashes)
        safe_name = l1_group.replace(" ", "_").replace("/", "-")
        out_path = fig_dir / f"level2_within_{safe_name}_{groupby_col}.pdf"
        plt.savefig(out_path)
        print(f"Saved: {out_path}")

        figs[l1_group] = (fig, ax)

    return figs


# Set random seed for reproducibility
seed_everything(19960915)

# Set variables
color = "level_2"

# Set directories
path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)
dir = path / "output/AIRSCAPE/"

fig_dir = dir / "celltype_composition"
fig_dir.mkdir(parents=True, exist_ok=True)

# Configure scanpy to save figures in our custom directory
sc.settings.figdir = fig_dir

# Set colors
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

# Load data
print("Loading data from 'adata_final_object/adata_with_metadata.zarr'...")
adata = ad.read_zarr(dir / "adata_final_object/adata_with_metadata.zarr")
print("Data loaded successfully.")

# --- Original level 2 composition plots ---
fig, ax = plot_celltype_composition(
    adata, celltype_col="level_2", groupby_col="condition"
)
fig, ax = plot_celltype_composition(adata, celltype_col="level_2", groupby_col="ROI")
fig, ax = plot_celltype_composition(
    adata, celltype_col="level_2", groupby_col="diagnosis"
)

# --- New: level 2 composition within each level 1 group ---
# One PDF per level 1 group, x-axis = diagnosis
figs = plot_level2_within_level1(
    adata,
    level1_col="level_1",
    level2_col="level_2",
    groupby_col="diagnosis",
)

# Also run for condition and ROI if needed:
figs = plot_level2_within_level1(adata, groupby_col="condition")
figs = plot_level2_within_level1(adata, groupby_col="ROI")

print("Composition plots generated and saved successfully.")
