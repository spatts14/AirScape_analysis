"""Generate plots for APT permutation results."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.airspace_colors import diagnosis_palette

ROI_ALIASES = {
    "IPF_RBH_15_OG": "IPF_RBH_15",
    "IPF_RBH_15_CORRECT": "IPF_RBH_15",
}


def corrected_roi_for_meta(roi_name: str) -> str:
    """Map known ROI aliases to the corrected metadata key."""
    return ROI_ALIASES.get(roi_name, roi_name)


# Define functions
def load_adjacency_p_values(input_dir: Path):
    """Load adjacency permutation test p-values for each ROI.

    Parameters:
        input_dir : Path
            Directory containing APT permutation test

    Returns:
        dict
            {roi_name: dataframe}

    """
    results = {}

    for file in sorted(
        input_dir.glob("nonfiltered_adjacency_permutation_test_p_values_*.csv")
    ):
        roi = file.stem.removeprefix("nonfiltered_adjacency_permutation_test_p_values_")
        df = pd.read_csv(file, index_col=0)

        results[roi] = df

    return results


def make_APT_celltype_dict(ROI_APT_dict, cell_type_list):
    """Combine APT dataframes for each cell type across conditions into a single dict.

    Args:
        ROI_APT_dict (dict): Dictionary of APT dataframes for each ROI
        cell_type_list (list): List of all cell types across conditions
    Returns:
        dict: Dictionary of combined APT dataframes for each cell type
    """
    celltype_dict = {}

    for cell_type in cell_type_list:
        combined = pd.DataFrame(
            {
                condition: df.loc[cell_type]
                if cell_type in df.index
                else pd.Series(float("nan"), index=df.columns)
                for condition, df in ROI_APT_dict.items()
            }
        )

        celltype_dict[cell_type] = combined

    return celltype_dict


def make_clustermap(
    cell_type, celltype_dict, meta, meta_cols, cmap="vlag", palette=None, fig_path=None
):
    """Make clustermap of APT scores for a given cell type.

    Args:
        cell_type (str): Cell type to plot
        celltype_dict (dict): Dictionary of dataframes for cell type across conditions
        meta (pd.DataFrame): Metadata dataframe with ROI information
        meta_cols (list): List of metadata columns to include in annotations
        cmap (str): Colormap for heatmap
        palette (dictionary): Dictionary of colors for annotation
        fig_path (Path): Path to save figure

    Returns:
        sns.ClusterGrid: ClusterGrid object containing the clustermap

    """
    safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

    # Build matrix
    celltype_df = pd.DataFrame(celltype_dict[cell_type])
    heatmap_df = celltype_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Validate metadata
    for col in meta_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing metadata column: {col}")

    meta_lookup = meta.copy()
    meta_lookup["_roi_meta_key"] = meta_lookup.index.map(corrected_roi_for_meta)
    meta_lookup = meta_lookup.set_index("_roi_meta_key")

    roi_lookup = [corrected_roi_for_meta(col) for col in heatmap_df.columns]
    meta_subset = meta_lookup.loc[roi_lookup, meta_cols].copy()
    meta_subset.index = heatmap_df.columns

    # Build color annotations + store palettes
    col_colors = pd.DataFrame(index=heatmap_df.columns)
    palettes = {}

    for col in meta_cols:
        # Unique values
        unique_vals = meta_subset[col].unique()

        # Number of unique values
        length = len(unique_vals)

        if palette is None:
            palette = dict(zip(unique_vals, sns.color_palette("husl", length)))

        # Store palette and map colors
        palettes[col] = palette
        col_colors[col] = meta_subset[col].map(palette)

    # Plot
    g = sns.clustermap(
        heatmap_df,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        figsize=(15, 8),
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=False,
        xticklabels=True,
        yticklabels=True,
    )

    g.figure.suptitle(f"{cell_type} - APT SES (p-val nonfiltered)", y=1.02)

    # Build legend (outside plot)
    legend_handles = []

    for col in meta_cols:
        for level, color in palettes[col].items():
            legend_handles.append(Patch(facecolor=color, label=f"{col}: {level}"))

    g.ax_heatmap.legend(
        handles=legend_handles,
        title="Metadata",
        bbox_to_anchor=(1.6, 1),
        loc="upper left",
        frameon=False,
    )

    # Save figure
    plt.savefig(
        fig_path / f"{safe_cell_type}_APT_SES_p_val_nonfiltered.pdf",
        bbox_inches="tight",
    )
    plt.close()


# Base project path
base_path = Path(
    "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

# Input
input_dir = (
    base_path
    / "output"
    / "muspan"
    / "adjacency_permutation_test_results"
    / "nonfiltered"
)

# Output directories
outpath = base_path / "output" / "muspan" / "APT_plots"
heatmap_path = outpath / "heatmaps_zscores"
barplot_path_zscore = outpath / "barplots_zscores"

# Create directories
for path in [outpath, heatmap_path, barplot_path_zscore]:
    path.mkdir(parents=True, exist_ok=True)

# Plot heatmap of z-scores for each condition
cmap = sns.color_palette("vlag", as_cmap=True)
set_palette = diagnosis_palette

meta_column = "diagnosis"

# Significance level for Mann-Whitney U test
alpha_level = 0.05

# Figure dir
fig_dir = outpath / "IPF_only"
fig_dir.mkdir(parents=True, exist_ok=True)


# Load metadata
meta = pd.read_csv(
    base_path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col=0
)

# Import adjacency permutation test
ROI_APT_dict = load_adjacency_p_values(input_dir)

# Get list of all cell types across conditions
cell_type_list = []
seen_cell_types = set()
for df in ROI_APT_dict.values():
    for cell_type in df.index.tolist():
        if cell_type not in seen_cell_types:
            seen_cell_types.add(cell_type)
            cell_type_list.append(cell_type)

# Combine dataframes for each cell type across conditions
celltype_dict = make_APT_celltype_dict(ROI_APT_dict, cell_type_list)

# Remove ROIs individually or as a group samples to focus on comparison of interest
for cell_type in cell_type_list:
    celltype_dict[cell_type] = celltype_dict[cell_type].drop(
        columns=[col for col in celltype_dict[cell_type].columns if "COPD" in col]
        + [col for col in celltype_dict[cell_type].columns if "MICA" in col]
        + [col for col in celltype_dict[cell_type].columns if "PM08" in col]
        # + [col for col in celltype_dict[cell_type].columns if "IPF_15" in col]
    )

# Get all cell types for plotting barplots
all_cell_types_list = list(celltype_dict.keys())

# Plot heatmaps and barplots for each cell type
for celltype_1 in all_cell_types_list:
    # Sanitize cell type name for file paths
    safe_cell_type_1 = celltype_1.replace("/", "_").replace(" ", "_")

    # HEATMAP OF APT Z-SCORES FOR ALL NEIGHBORING CELL TYPES
    # Heatmap data
    celltype_df = pd.DataFrame(celltype_dict[celltype_1])
    heatmap_df = celltype_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Metadata aligned to heatmap columns
    meta_subset = meta.copy()
    meta_subset["_roi_meta_key"] = meta_subset.index.map(corrected_roi_for_meta)
    meta_subset = meta_subset.set_index("_roi_meta_key")
    meta_subset = meta_subset.loc[
        [corrected_roi_for_meta(col) for col in heatmap_df.columns], meta_column
    ]
    meta_subset.index = heatmap_df.columns

    # Column annotation colors
    col_colors = meta_subset.map(set_palette)

    # Plot
    g = sns.clustermap(
        heatmap_df,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        figsize=(10, 8),
        col_colors=col_colors,
        xticklabels=True,
        yticklabels=True,
        col_cluster=False,
        vmax=75,  # set max for color scale to see what is interesting
    )

    g.figure.suptitle(f"{celltype_1} - APT SES (p-val nonfiltered)", y=0.85)

    # Set cbar label and move it to the right of the plot
    g.ax_cbar.set_position([-0.07, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    g.ax_cbar.set_ylabel("SES (p-val nonfiltered)", rotation=90, labelpad=-60)

    plt.savefig(
        fig_dir / f"{safe_cell_type_1}_APT_SES_p_val_nonfiltered_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close()
