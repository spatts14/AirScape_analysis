"""Generate plots for APT permutation results."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import ttest_ind, ttest_rel

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.airspace_colors import time_treatment_arm_palette, treatment_arm_palette

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


def calc_change_BL(df):
    """Pivot to one row per donor with BASELINE/6WK/6MO columns, then compute change scores."""
    pivot = df.pivot_table(
        index=["sample_ID", "treatment_arm"],
        columns="time_point_label",
        values="SES (p-val nonfiltered)",
        aggfunc="first",
    ).reset_index()

    for col in ["BASELINE", "6 WEEKS", "6 MONTHS"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["change_BL"] = pivot["BASELINE"] - pivot["BASELINE"]
    pivot["change_BL_6W"] = pivot["6 WEEKS"] - pivot["BASELINE"]
    pivot["change_BL_6M"] = pivot["6 MONTHS"] - pivot["BASELINE"]

    return pivot[
        [
            "sample_ID",
            "treatment_arm",
            "BASELINE",
            "6 WEEKS",
            "6 MONTHS",
            "change_BL",
            "change_BL_6W",
            "change_BL_6M",
        ]
    ]


def calc_stats(change_df, change_df_long):
    """Compute the four requested comparisons.

    Paired tests use change_df (wide, one row per donor) so that BASELINE/
    6 WEEKS/6 MONTHS values for the *same* donor line up in the same row.
    NaNs (donor missing that timepoint) are dropped pairwise before testing.

    Unpaired tests use change_df_long, pulling out the relevant
    treatment_arm/time_point_label subsets as independent samples.
    """
    results = {}

    # --- Paired: TREATMENT baseline vs TREATMENT 6 weeks ---
    treat = change_df[change_df["treatment_arm"] == "TREATMENT"]
    paired_bl_6w = treat[["BASELINE", "6 WEEKS"]].dropna()
    t_stat, p_val = ttest_rel(paired_bl_6w["BASELINE"], paired_bl_6w["6 WEEKS"])
    results["TREATMENT: BASELINE vs 6 WEEKS"] = {
        "test": "paired",
        "n": len(paired_bl_6w),
        "t_stat": t_stat,
        "p_value": float(f"{p_val:.3f}"),
    }

    # --- Paired: TREATMENT baseline vs TREATMENT 6 months ---
    paired_bl_6m = treat[["BASELINE", "6 MONTHS"]].dropna()
    t_stat, p_val = ttest_rel(paired_bl_6m["BASELINE"], paired_bl_6m["6 MONTHS"])
    results["TREATMENT: BASELINE vs 6 MONTHS"] = {
        "test": "paired",
        "n": len(paired_bl_6m),
        "t_stat": t_stat,
        "p_value": float(f"{p_val:.3f}"),
    }

    # --- Paired: TREATMENT 6 weeks vs TREATMENT 6 months ---
    paired_6w_6m = treat[["6 WEEKS", "6 MONTHS"]].dropna()
    t_stat, p_val = ttest_rel(paired_6w_6m["6 WEEKS"], paired_6w_6m["6 MONTHS"])
    results["TREATMENT: 6 WEEKS vs 6 MONTHS"] = {
        "test": "paired",
        "n": len(paired_6w_6m),
        "t_stat": t_stat,
        "p_value": float(f"{p_val:.3f}"),
    }

    # --- Unpaired: SHAM 6 weeks vs TREATMENT 6 weeks ---
    sham_6w = change_df_long[
        (change_df_long["treatment_arm"] == "SHAM")
        & (change_df_long["time_point_label"] == "change_BL_6W")
    ]["change_from_baseline"].dropna()

    treat_6w = change_df_long[
        (change_df_long["treatment_arm"] == "TREATMENT")
        & (change_df_long["time_point_label"] == "change_BL_6W")
    ]["change_from_baseline"].dropna()

    t_stat, p_val = ttest_ind(sham_6w, treat_6w, equal_var=False)  # Welch's t-test
    results["SHAM 6 WEEKS vs TREATMENT 6 WEEKS"] = {
        "test": "unpaired",
        "n1": len(sham_6w),
        "n2": len(treat_6w),
        "t_stat": t_stat,
        "p_value": float(f"{p_val:.3f}"),
    }

    # --- Unpaired: SHAM 6 months vs TREATMENT 6 months ---
    sham_6m = change_df_long[
        (change_df_long["treatment_arm"] == "SHAM")
        & (change_df_long["time_point_label"] == "change_BL_6M")
    ]["change_from_baseline"].dropna()

    treat_6m = change_df_long[
        (change_df_long["treatment_arm"] == "TREATMENT")
        & (change_df_long["time_point_label"] == "change_BL_6M")
    ]["change_from_baseline"].dropna()

    t_stat, p_val = ttest_ind(sham_6m, treat_6m, equal_var=False)  # Welch's t-test
    results["SHAM 6 MONTHS vs TREATMENT 6 MONTHS"] = {
        "test": "unpaired",
        "n1": len(sham_6m),
        "n2": len(treat_6m),
        "t_stat": t_stat,
        "p_value": float(f"{p_val:.3f}"),
    }

    return results


def p_to_asterisks(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def add_brackets(ax, comparisons, base_y, step, fontsize=11):
    """
    comparisons = [
        (x1, x2, pval, color),
        ...
    ]

    x1, x2 MUST be categorical x positions (not dodged positions).
    """

    comparisons = sorted(
        comparisons,
        key=lambda x: abs(x[1] - x[0]),
    )

    for i, (x1, x2, pval, color) in enumerate(comparisons):
        y = base_y + i * step * 1.5
        h = step * 0.8

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=1.2,
            color=color,
            clip_on=False,
            zorder=100,
        )

        ax.text(
            (x1 + x2) / 2,
            y + h,
            p_to_asterisks(pval),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            clip_on=False,
            zorder=101,
        )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, base_y + len(comparisons) * step * 1.8)


def get_category_positions(ax):
    """
    Returns mapping: category label -> x coordinate used by seaborn.
    """
    return {tick.get_text(): tick.get_position()[0] for tick in ax.get_xticklabels()}


def get_dodged_category_positions(ax, hue_order):
    """Return x positions for each category and hue level."""

    centers = get_category_positions(ax)
    hue_count = len(hue_order)
    if hue_count < 1:
        raise ValueError("hue_order must contain at least one level")

    dodge_width = 0.8
    if hue_count == 1:
        offsets = [0.0]
    else:
        step = dodge_width / hue_count
        start = -dodge_width / 2 + step / 2
        offsets = [start + i * step for i in range(hue_count)]

    return {
        category: {
            hue: centers[category] + offset for hue, offset in zip(hue_order, offsets)
        }
        for category in centers
    }


def make_plot(
    plot_df,
    stats_plot_dir,
    celltype_1,
    cell_type_2,
    safe_cell_type_1,
    safe_cell_type_2,
    meta_column,
    treatment_arm_palette_list,
):
    # Calculate change from baseline for each donor (wide format, needed for paired tests)
    change_df = calc_change_BL(plot_df)

    # Put into long format for plotting
    change_df_long = change_df.melt(
        id_vars=["sample_ID", "treatment_arm"],
        value_vars=["change_BL", "change_BL_6W", "change_BL_6M"],
        var_name="time_point_label",
        value_name="change_from_baseline",
    )

    # --- Run all four statistical tests ---
    stats_results = calc_stats(change_df, change_df_long)

    for label, res in stats_results.items():
        print(f"{label} ({res['test']}): t={res['t_stat']:.3f}, p={res['p_value']:.4g}")

    # Box plot
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.stripplot(
        data=change_df_long,
        x="time_point_label",
        y="change_from_baseline",
        hue="treatment_arm",
        hue_order=["SHAM", "TREATMENT"],
        dodge=True,
        alpha=1,
        linewidth=0.5,
        palette=treatment_arm_palette_list,
        ax=ax,
    )

    sns.boxenplot(
        data=change_df_long,
        x="time_point_label",
        y="change_from_baseline",
        hue="treatment_arm",
        hue_order=["SHAM", "TREATMENT"],
        dodge=True,
        alpha=0.5,
        palette=treatment_arm_palette_list,
        ax=ax,
    )

    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    n = len(plot_df[meta_column].unique())
    ax.legend(
        handles[:n],
        labels[:n],
        title=meta_column,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # ------------------------------------------------------------------
    # Add significance brackets
    # ------------------------------------------------------------------
    # --- TRUE seaborn category positions (no guessing) ---
    x_pos = get_dodged_category_positions(ax, ["SHAM", "TREATMENT"])

    # Extract p-values
    p1 = stats_results["TREATMENT: BASELINE vs 6 WEEKS"]["p_value"]
    p2 = stats_results["TREATMENT: BASELINE vs 6 MONTHS"]["p_value"]
    p3 = stats_results["TREATMENT: 6 WEEKS vs 6 MONTHS"]["p_value"]
    p4 = stats_results["SHAM 6 WEEKS vs TREATMENT 6 WEEKS"]["p_value"]
    p5 = stats_results["SHAM 6 MONTHS vs TREATMENT 6 MONTHS"]["p_value"]

    # --- Build comparisons using category centers ONLY ---
    comparisons = [
        (
            x_pos["change_BL"]["TREATMENT"],
            x_pos["change_BL_6W"]["TREATMENT"],
            p1,
            "black",
        ),
        (
            x_pos["change_BL"]["TREATMENT"],
            x_pos["change_BL_6M"]["TREATMENT"],
            p2,
            "black",
        ),
        (
            x_pos["change_BL_6W"]["TREATMENT"],
            x_pos["change_BL_6M"]["TREATMENT"],
            p3,
            "black",
        ),
        (
            x_pos["change_BL_6W"]["SHAM"],
            x_pos["change_BL_6W"]["TREATMENT"],
            p4,
            "grey",
        ),
        (
            x_pos["change_BL_6M"]["SHAM"],
            x_pos["change_BL_6M"]["TREATMENT"],
            p5,
            "grey",
        ),
    ]

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    base_y = ymax + (y_range * 0.08 if y_range else 1.0)
    step = y_range * 0.12 if y_range else 1.0

    add_brackets(
        ax,
        comparisons,
        base_y=base_y,
        step=step,
    )
    # Set labels and title
    ax.set_xticklabels(["Baseline", "6 Weeks", "6 Months"])

    plt.title(f"{celltype_1} vs {cell_type_2}", fontsize=14)
    plt.xlabel("", fontsize=14)
    plt.ylabel("SES (normalized to baseline)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        stats_plot_dir
        / f"{safe_cell_type_1}_{safe_cell_type_2}_{meta_column}_STAT_CHANGE_FROM_BASELINE_SES_p_val.pdf"
    )
    plt.close()

    return stats_results


def is_valid_for_analysis(df, min_n=3):
    required = [
        "BASELINE SHAM",
        "BASELINE TREATMENT",
        "6 WEEKS SHAM",
        "6 WEEKS TREATMENT",
        "6 MONTHS SHAM",
        "6 MONTHS TREATMENT",
    ]

    counts = df.groupby("time_treatment_arm")["SES (p-val nonfiltered)"].count()

    for r in required:
        if counts.get(r, 0) < min_n:
            return False

    return True


# # Base project path
base_path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)
# base_path = Path(
#     "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
# )

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
time_treatment_arm_palette_list = list(time_treatment_arm_palette.values())
treatment_arm_palette_list = list(treatment_arm_palette.values())

# Set parameters for plotting
meta_column = "time_treatment_arm"

# Desired diagnosis order
meta_column_order = [
    "BASELINE SHAM",
    "BASELINE TREATMENT",
    "6 WEEKS SHAM",
    "6 WEEKS TREATMENT",
    "6 MONTHS SHAM",
    "6 MONTHS TREATMENT",
]

# Significance level for Mann-Whitney U test
alpha_level = 0.05

# Figure dir
fig_dir = outpath / "COPD"
fig_dir.mkdir(parents=True, exist_ok=True)

# Make stats dir
stats_dir = fig_dir / "_STATS"
stats_dir.mkdir(parents=True, exist_ok=True)

# Make main plot output dir
stats_plot_dir = fig_dir / "_STATS_PLOTS"
stats_plot_dir.mkdir(parents=True, exist_ok=True)

# Load metadata
meta = pd.read_csv(
    base_path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col=0
)

# Add column for timepoint and treatment arm
meta["time_treatment_arm"] = meta["time_point_label"] + " " + meta["treatment_arm"]

# Set color palette for plots
set_palette = time_treatment_arm_palette

# Check the color key
missing_palette_keys = [key for key in meta_column_order if key not in set_palette]
if missing_palette_keys:
    raise ValueError(f"Missing palette keys: {missing_palette_keys}")

# Import adjacency permutation test
ROI_APT_dict = load_adjacency_p_values(input_dir)

# Get list of all cell types across conditions
cell_type_list = next(iter(ROI_APT_dict.values())).index.tolist()

# Combine dataframes for each cell type across conditions
celltype_dict = make_APT_celltype_dict(ROI_APT_dict, cell_type_list)

# Remove ROIs individually or as a group samples to focus on comparison of interest
for cell_type in cell_type_list:
    celltype_dict[cell_type] = celltype_dict[cell_type].drop(
        columns=[col for col in celltype_dict[cell_type].columns if "IPF" in col]
        + [col for col in celltype_dict[cell_type].columns if "PM08" in col]
        + [col for col in celltype_dict[cell_type].columns if "MICA" in col]
        + [col for col in celltype_dict[cell_type].columns if "COPD_R010_V2" in col]
    )

# Get all cell types for plotting barplots
all_cell_types_list = list(celltype_dict.keys())

stats_results_dict = {}

# Plot heatmaps and barplots for each cell type
for celltype_1 in all_cell_types_list:
    # Sanitize cell type name for file paths
    safe_cell_type_1 = celltype_1.replace("/", "_").replace(" ", "_")

    # HEATMAP OF APT Z-SCORES FOR ALL NEIGHBORING CELL TYPES

    # Make directory for heatmaps if it doesn't exist
    fig_dir_heatmap = fig_dir / "heatmaps"
    fig_dir_heatmap.mkdir(parents=True, exist_ok=True)

    # Build heatmap dataframe for this cell type
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

    # Order samples by diagnosis
    ordered_columns = (
        meta_subset.map(
            lambda x: meta_column_order.index(x)
            if x in meta_column_order
            else len(meta_column_order)
        )
        .sort_values()
        .index
    )

    heatmap_df = heatmap_df[ordered_columns]
    meta_subset = meta_subset.loc[ordered_columns]

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
    )

    g.figure.suptitle(f"{celltype_1} - APT SES (p-val nonfiltered)", y=0.85)

    # Set cbar label and move it to the right of the plot
    g.ax_cbar.set_position([-0.07, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    g.ax_cbar.set_ylabel("SES (p-val nonfiltered)", rotation=90, labelpad=-60)

    # Legend
    legend_handles = [
        Patch(facecolor=set_palette[diag], label=diag)
        for diag in meta_column_order
        if diag in set_palette
    ]

    g.ax_heatmap.legend(
        handles=legend_handles,
        title=meta_column,
        bbox_to_anchor=(1.5, 1),
        loc="upper left",
        frameon=False,
    )

    plt.savefig(
        fig_dir_heatmap / f"{safe_cell_type_1}_APT_SES_p_val_nonfiltered_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # BOX PLOTS OF APT Z-SCORES FOR EACH NEIGHBORING CELL TYPE, FACETTED BY DIAGNOSIS
    for cell_type_2 in all_cell_types_list:
        # Make cell_type_2_of_interest safe for file paths
        safe_cell_type_2 = cell_type_2.replace("/", "_").replace(" ", "_")

        if meta_column not in meta.columns:
            raise ValueError(
                f"Metadata column '{meta_column}' not found in meta dataframe"
            )

        # make directory for this cell type if it doesn't exist
        celltype_dir = (
            fig_dir
            / "cellcomparison_boxplots"
            / f"{safe_cell_type_1}_{safe_cell_type_2}"
        )
        celltype_dir.mkdir(parents=True, exist_ok=True)

        # Extract the dataframe for this cell type across conditions
        celltype_df = pd.DataFrame(celltype_dict[celltype_1])

        # SUBSET ON CELL TYPE 2 OF INTEREST
        celltype_df = celltype_df.loc[[cell_type_2]]

        # Make into dataframe
        df_groups = pd.DataFrame(celltype_df)

        # Transpose and melt the dataframe for plotting
        plot_df = celltype_df.transpose().reset_index().rename(columns={"index": "ROI"})

        plot_df = plot_df.melt(
            id_vars="ROI",
            var_name="Neighbor Cell Type",
            value_name="SES (p-val nonfiltered)",
        )

        plot_df["ROI_meta_key"] = plot_df["ROI"].map(corrected_roi_for_meta)

        # # Add metadata columns by extracting from ROI name
        plot_df = plot_df.merge(meta, left_on="ROI_meta_key", right_index=True)
        plot_df = plot_df.drop(columns=["ROI_meta_key"])

        # Drop NaN values before plotting
        plot_df = plot_df.dropna(subset=["SES (p-val nonfiltered)"])

        # Ensure sufficient observations for plotting and statistical analysis
        if not is_valid_for_analysis(plot_df):
            print(
                f"Skipping {celltype_1} vs {cell_type_2}: "
                f"insufficient observations per time_treatment_arm"
            )
            continue

        # Set order of conditions for plotting if diagnosis column is present
        if meta_column in plot_df.columns:
            plot_df[meta_column] = pd.Categorical(
                plot_df[meta_column],
                categories=meta_column_order,
                ordered=True,
            )

        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.stripplot(
            data=plot_df,
            x="Neighbor Cell Type",
            y="SES (p-val nonfiltered)",
            hue=meta_column,
            dodge=True,
            alpha=1,
            linewidth=0.5,
            palette=set_palette,
            ax=ax,
        )

        sns.boxenplot(
            data=plot_df,
            x="Neighbor Cell Type",
            y="SES (p-val nonfiltered)",
            hue=meta_column,
            dodge=True,
            alpha=0.5,
            palette=set_palette,
            ax=ax,
        )

        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        n = len(plot_df[meta_column].unique())

        ax.legend(
            handles[:n],
            labels[:n],
            title=meta_column,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

        # ax.set_title(f"{celltype_1}\nstat={stat}, p={p_value:.3f}", fontsize=14)
        ax.set_title(f"{celltype_1}", fontsize=14)
        ax.set_xticklabels([""])
        ax.set_xlabel(cell_type_2, fontsize=14)
        ax.set_ylabel("SES", fontsize=14)

        plt.tight_layout()
        plt.savefig(
            celltype_dir
            / f"{safe_cell_type_1}_{safe_cell_type_2}_{meta_column}_SES_p_val_nonfiltered.pdf"
        )
        plt.close()

        # Set plot
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.set_title(f"{celltype_1} vs {cell_type_2}", fontsize=14)
        # Individual donor lines, colored by arm
        sns.lineplot(
            data=plot_df,
            x="time_point_label",
            y="SES (p-val nonfiltered)",
            hue="treatment_arm",
            hue_order=["SHAM", "TREATMENT"],
            units="sample_ID",
            estimator=None,  # one line per donor, no aggregation
            marker="o",
            palette=treatment_arm_palette_list,
            ax=ax,
        )
        ax.set_ylabel("SES", fontsize=14)

        plt.tight_layout()
        plt.savefig(
            celltype_dir
            / f"TREATMENT_SHAM_{safe_cell_type_1}_{safe_cell_type_2}_{meta_column}_SES_p_val_nonfiltered.pdf"
        )

        plt.close()

        # CALCULATE FROM BASELINE CHANGE FOR EACH DONOR, PLOT AS STRIP/BOXEN AND LINE PLOTS

        # Plot change from baseline for each donor, colored by arm

        # Calculate change from baseline for each donor
        change_df = calc_change_BL(plot_df)

        # Put into long format for plotting
        change_df_long = change_df.melt(
            id_vars=["sample_ID", "treatment_arm"],
            value_vars=["change_BL", "change_BL_6W", "change_BL_6M"],
            var_name="time_point_label",
            value_name="change_from_baseline",
        )

        # Plot with stats
        stats_results = make_plot(
            plot_df=plot_df,
            stats_plot_dir=stats_plot_dir,
            celltype_1=celltype_1,
            cell_type_2=cell_type_2,
            safe_cell_type_1=safe_cell_type_1,
            safe_cell_type_2=safe_cell_type_2,
            meta_column=meta_column,
            treatment_arm_palette_list=treatment_arm_palette_list,
        )

        # Store stats results in dictionary for later export
        stats_results_dict[(celltype_1, cell_type_2)] = stats_results

        # Box plot (without stats brackets)
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.stripplot(
            data=change_df_long,
            x="time_point_label",
            y="change_from_baseline",
            hue="treatment_arm",
            hue_order=["SHAM", "TREATMENT"],
            dodge=True,
            alpha=1,
            linewidth=0.5,
            palette=treatment_arm_palette_list,
            ax=ax,
        )

        sns.boxenplot(
            data=change_df_long,
            x="time_point_label",
            y="change_from_baseline",
            hue="treatment_arm",
            hue_order=["SHAM", "TREATMENT"],
            dodge=True,
            alpha=0.5,
            palette=treatment_arm_palette_list,
            ax=ax,
        )

        plt.title(f"{celltype_1} vs {cell_type_2}", fontsize=14)
        plt.xlabel(" ", fontsize=14)
        plt.ylabel("SES (normalized to baseline)", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            celltype_dir
            / f"CHANGE_FROM_BASELINE_{safe_cell_type_1}_{safe_cell_type_2}_{meta_column}_SES_p_val_nonfiltered.pdf"
        )
        plt.close()

        ## Line plot (without stats brackets)
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.lineplot(
            data=change_df_long,
            x="time_point_label",
            y="change_from_baseline",
            hue="treatment_arm",
            hue_order=["SHAM", "TREATMENT"],
            units="sample_ID",
            palette=treatment_arm_palette_list,
            estimator=None,
            marker="o",
        )
        plt.axhline(0, color="gray", linestyle="--")

        plt.title(f"{celltype_1} vs {cell_type_2}", fontsize=14)
        plt.xlabel(" ", fontsize=14)
        plt.ylabel("SES (normalized to baseline)", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            celltype_dir
            / f"LINEPLOT_CHANGE_FROM_BASELINE_{safe_cell_type_1}_{safe_cell_type_2}_{meta_column}_SES_p_val_nonfiltered.pdf"
        )

        plt.close()

        print(f"Completed plots for {celltype_1} vs {cell_type_2}.")

# Export stats results to CSV
stats_results_df = pd.DataFrame.from_dict(
    {(i, j): stats_results_dict[(i, j)] for i, j in stats_results_dict.keys()},
    orient="index",
)
stats_results_df.to_csv(stats_dir / "APT_stats_results.csv")
stats_results_df.to_excel(stats_dir / "APT_stats_results.xlsx")

print(f"All plots and stats results saved to {fig_dir} and {stats_dir}.")
print("Script completed successfully.")
