"""Generate composition of celltype plots."""

import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from pandas import pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.seed_everything import seed_everything


def plot_celltype_composition(
    adata,
    fig_dir: Path,
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
        fig_dir : Path
            Directory to save the figure

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

    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")

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


# Mapping of level 1 group name -> the level 2 subtypes that belong to it.
LEVEL1_TO_LEVEL2 = {
    "Airway epithelial cells": [
        "Ciliated cells",
        "Goblet cells",
        "Basal cells",
        "Secretory epithelial cells",
        "Proliferating Basal cells",
    ],
    "Alveolar epithelial cells": [
        "Unknown",
        "AT2 cells",
        "AT1 cells",
        "Proliferating AT2 cells",
    ],
    "Endothelial cells": [
        "Unknown",
        "Blood endothelial cells",
        "Capillary endothelial cells",
        "Pulmonary artery endothelial cells",
        "Pulmonary vein endothelial cells",
        "Lymphatic endothelial cells",
        "Aerocytes",
    ],
    "Immune cells": [
        "Mast cells",
        "Plasma cells",
        "Unknown",
        "T cells",
        "CD4+ T cells",
        "CD8+ T cells",
        "Lymphocytes",
        "Interstitial macrophages",
        "Lipid-associated macrophages",
        "Monocytes/Neutrophils",
        "Dendritic cells",
        "B cells",
        "Airway/Alveolar macrophages",
        "Natural killer cells",
    ],
    "Stromal cells": [
        "Pericytes",
        "SMC",
        "CTHRC1+ fibroblasts",
        "Unknown",
        "Adventitial fibroblasts",
        "Alveolar fibroblasts",
        "Alveolar fibroblasts - collagen hi",
    ],
}


def plot_level2_within_level1(
    adata,
    fig_dir: Path,
    level1_col: str = "level_1",
    level2_col: str = "level_2",
    groupby_col: str = "diagnosis",
    figsize: tuple = (10, 6),
    palette: str = "tab20",
    ylabel: str = "Percentage (%)",
    level1_to_level2: dict[str, list[str]] | None = None,
):
    """Plot stacked bar charts showing level 2 composition within each level 1 group.

    The x-axis of each plot is groupby_col (e.g. diagnosis). Bar segments show
    what fraction each level 2 subtype makes up of that level 1 group's total
    cells. Only the level 2 subtypes defined in level1_to_level2 are included
    for each group, ensuring cross-contaminating labels don't inflate results.

    For example, for level 1 = "Airway epithelium", the plot shows what
    percentage of airway cells are Goblet, Ciliated, Basal, etc., broken down
    by diagnosis.

    Args:
        adata : AnnData
            Annotated data object with obs columns for level 1, level 2, and groupby.
        fig_dir : Path
            Directory to save the figures.
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
        level1_to_level2 : dict, optional
            Mapping of level 1 group name -> list of valid level 2 subtypes.
            Only the listed level 2 subtypes will be included for each level 1
            group's plot. Defaults to the module-level LEVEL1_TO_LEVEL2 dict.
            Pass an empty dict {} to disable filtering and plot all level 2
            labels present in the data.

    Returns:
        figs : dict
            Dictionary mapping level 1 group name -> (fig, ax) tuple.
    """
    # Validate columns
    for col in [level1_col, level2_col, groupby_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    # Resolve the level1 -> level2 whitelist
    if level1_to_level2 is None:
        level1_to_level2 = LEVEL1_TO_LEVEL2

    # Build a color dict for all level 2 cell types up front so colours are
    # consistent across all per-level-1 plots
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

        # Apply the level 2 whitelist for this level 1 group (if defined)
        valid_level2 = level1_to_level2.get(l1_group, None)
        if valid_level2 is not None:
            before = subset.shape[0]
            subset = subset[subset[level2_col].isin(valid_level2)]
            dropped = before - subset.shape[0]
            if dropped > 0:
                print(
                    f"  '{l1_group}': excluded {dropped} cells with level 2 labels "
                    f"not in the whitelist."
                )
            if subset.shape[0] == 0:
                print(f"Skipping '{l1_group}': no cells remain after filtering.")
                continue
        else:
            print(
                f"  '{l1_group}': no whitelist entry found — "
                f"plotting all level 2 labels present in the data."
            )

        # Count level 2 subtypes within each groupby value
        counts = (
            subset.groupby([groupby_col, level2_col], observed=True)
            .size()
            .reset_index(name="count")
        )

        # Keep only level 2 subtypes that have at least one cell in this subset
        level2_present = counts.loc[counts["count"] > 0, level2_col].unique().tolist()
        counts = counts[counts[level2_col].isin(level2_present)]

        # Percentage: denominator is total cells in this level 1 group per
        # groupby value (i.e. after whitelist filtering)
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

        ax.set_xticks(x_positions)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")

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

        safe_name = l1_group.replace(" ", "_").replace("/", "-")
        out_path = fig_dir / f"level2_within_{safe_name}_{groupby_col}.pdf"
        plt.savefig(out_path)
        print(f"  Saved: {out_path}")

        figs[l1_group] = (fig, ax)

    return figs


def main():
    """Main function to generate cell type composition plots."""
    # Set random seed for reproducibility
    seed_everything(19960915)

    # Set directories
    path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
    )
    dir = path / "output/AIRSCAPE/"

    fig_dir = dir / "celltype_composition"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sc.settings.figdir = fig_dir

    # Load data
    print("Loading data from 'adata_final_object/adata_with_metadata.zarr'...")
    adata = ad.read_zarr(dir / "adata_final_object/adata_with_metadata.zarr")
    print("Data loaded successfully.")

    # Subset on conditions of interest
    conditions_of_interest = ["COPD", "MICA"]
    adata = adata[adata.obs["condition"].isin(conditions_of_interest)].copy()

    if conditions_of_interest == ["COPD", "MICA"]:
        # Add a new column to adata.obs that combines treatment arm and timepoint
        adata.obs["time_treatment_arm"] = (
            adata.obs["treatment_arm"] + " " + adata.obs["time_point_label"]
        )

        # Set order for time_treatment_arm
        time_treatment_order = [
            "SHAM BASELINE",
            "SHAM 6 WEEKS",
            "SHAM 6 MONTHS",
            "TREATMENT BASELINE",
            "TREATMENT 6 WEEKS",
            "TREATMENT 6 MONTHS",
        ]
        adata.obs["time_treatment_arm"] = pd.Categorical(
            adata.obs["time_treatment_arm"],
            categories=time_treatment_order,
            ordered=True,
        )

    # Save to subset df
    if conditions_of_interest:
        subset_conditions = "v".join(conditions_of_interest)
        fig_dir = fig_dir / f"{subset_conditions}"
        fig_dir.mkdir(parents=True, exist_ok=True)
    else:
        fig_dir = fig_dir / "all"
        fig_dir.mkdir(parents=True, exist_ok=True)

    # Original level 2 composition plots
    plot_celltype_composition(
        adata, fig_dir=fig_dir, celltype_col="level_2", groupby_col="condition"
    )
    plot_celltype_composition(
        adata, fig_dir=fig_dir, celltype_col="level_2", groupby_col="ROI"
    )
    plot_celltype_composition(
        adata, fig_dir=fig_dir, celltype_col="level_2", groupby_col="diagnosis"
    )

    # New: level 2 composition within each level 1 group
    # One PDF per level 1 group; only the whitelisted level 2 subtypes are shown.
    plot_level2_within_level1(
        adata,
        fig_dir=fig_dir,
        level1_col="level_1",
        level2_col="level_2",
        groupby_col="diagnosis",
    )
    plot_level2_within_level1(adata, fig_dir=fig_dir, groupby_col="condition")
    plot_level2_within_level1(adata, fig_dir=fig_dir, groupby_col="ROI")

    if conditions_of_interest == ["COPD", "MICA"]:
        print("Generating additional composition plots for COPD vs MICA conditions...")
        plot_celltype_composition(
            adata,
            fig_dir=fig_dir,
            celltype_col="level_2",
            groupby_col="time_treatment_arm",
        )
        plot_level2_within_level1(
            adata, fig_dir=fig_dir, groupby_col="time_treatment_arm"
        )

    print("Composition plots generated and saved successfully.")


if __name__ == "__main__":
    main()
