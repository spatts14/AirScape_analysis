"""Plots for dotplots and boxplotsls."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def plot_gene_expression_per_roi(
    adata,
    gene,
    roi_col="ROI",
    condition_col="condition",
    plot_col="timepoint",
    hue_col=None,
    agg_func="mean",
    layer="counts",
    new_fig_dir=None,
):
    """Plot ROI-level expression for a given gene.

    Args:
    adata : AnnData
        AnnData object containing expression matrix.
    gene : str
        Gene name to plot.
    roi_col : str
        Column in adata.obs defining ROI.
    condition_col : str
        Column in adata.obs defining condition.
    plot_col : str
        Column in adata.obs to use for x-axis grouping (e.g. timepoint).
    hue_col : str or None
        Column in adata.obs to use for color grouping (e.g. treatment arm). Optional.
    agg_func : str
        Aggregation across spots within ROI ('sum', 'mean', etc).
    layer : str or None
        Layer in adata.layers to use for expression values (default: "counts").
    new_fig_dir : str or None
        Subdirectory within fig_dir to save the plot (default: None).

    Returns:
    plot_df : pd.DataFrame
        Aggregated dataframe used for plotting.
    """
    base_color_dict = {
        "Treatment": "#9499B1",
        "Sham": "#9CA9A0",
        "None": "#E6D6A3",
    }

    # Validate inputs
    required_cols = [roi_col, condition_col, plot_col]
    if hue_col:
        required_cols.append(hue_col)

    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")

    if gene not in adata.var_names:
        raise ValueError(f"{gene} not found in adata.var_names")

    # Extract expression
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        expr = adata[:, gene].layers[layer]
    else:
        expr = adata[:, gene].X

    # Convert to dense 1D array
    if hasattr(expr, "toarray"):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()

    # Build dataframe
    temp_df = pd.DataFrame(
        {
            f"{gene}_expression": expr,
            roi_col: adata.obs[roi_col].values,
            condition_col: adata.obs[condition_col].values,
            plot_col: adata.obs[plot_col].values,
        }
    )

    if hue_col:
        temp_df[hue_col] = adata.obs[hue_col].values

    # Aggregate per ROI
    groupby_cols = [roi_col, condition_col, plot_col]
    if hue_col:
        groupby_cols.append(hue_col)

    plot_df = (
        temp_df.groupby(groupby_cols, as_index=False, observed=True)
        .agg({f"{gene}_expression": agg_func})
        .rename(columns={f"{gene}_expression": f"{gene}_aggregated"})
    )

    print(f"ROI-level {gene} values:")
    print(plot_df.head())

    # Handle hue safely
    if hue_col:
        plot_df[hue_col] = plot_df[hue_col].astype(object).fillna("None")

        hue_levels = plot_df[hue_col].unique().tolist()

        missing_levels = [level for level in hue_levels if level not in base_color_dict]

        if missing_levels:
            fallback_colors = sns.color_palette("tab10b", n_colors=len(missing_levels))
            for level, color in zip(missing_levels, fallback_colors):
                base_color_dict[level] = color

        hue_palette = {level: base_color_dict[level] for level in hue_levels}

    # Plot
    plt.figure(figsize=(6, 6))

    sns.boxplot(
        data=plot_df,
        x=plot_col,
        y=f"{gene}_aggregated",
        hue=hue_col if hue_col else None,
        hue_order=["None", "Sham", "Treatment"],
        palette=hue_palette if hue_col else None,
        showfliers=False,
    )

    sns.stripplot(
        data=plot_df,
        x=plot_col,
        y=f"{gene}_aggregated",
        hue=hue_col if hue_col else None,
        hue_order=["None", "Sham", "Treatment"],
        dodge=True if hue_col else False,
        color="black",
        size=4,
        alpha=0.6,
    )

    # Fix duplicated legends
    if hue_col:
        handles, labels = plt.gca().get_legend_handles_labels()
        n = len(hue_levels)
        plt.legend(handles[:n], labels[:n], title=hue_col)
    else:
        plt.legend().set_visible(False)

    plt.xlabel(plot_col)
    plt.ylabel(f"{gene} expression ({agg_func})")
    plt.title(f"{gene} expression per ROI by {plot_col}")

    plt.tight_layout()

    # Save
    if new_fig_dir:
        save_dir = fig_dir / new_fig_dir
    else:
        save_dir = fig_dir

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_dir / f"{gene}_expression_per_ROI_{plot_col}.pdf")
    plt.close()

    return plot_df


def plot_gene_dotplot(adata, gene, level, meta, cmap, fig_dir, save_name=None):
    """Create a dot plot of gene expression across groups.

    Args:
    adata : AnnData
        AnnData object
    gene : str
        Gene name (must be in adata.var_names)
    level : str
        Column in adata.obs for y-axis grouping (e.g. cell type)
    meta : str
        Column in adata.obs for x-axis grouping (e.g. condition/timepoint)
    cmap : matplotlib colormap
        Colormap for expression values
    fig_dir : pathlib.Path or str
        Directory to save the figure
    save_name : str, optional
        Name for the saved figure (default: None)
    """
    # Check for required columns
    required_cols = [level, meta]
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing columns in adata.obs: {missing}")

    if gene not in adata.var_names:
        raise ValueError(f"{gene} not found in adata.var_names")

    # Compute per-group stats
    df = adata.obs[required_cols].copy()

    X = adata[:, gene].X  # SHOULD I BE USING COUNTS LAYER HERE? OR NORMALISED?
    if hasattr(X, "toarray"):  # sparse
        expr = X.toarray().flatten()
    else:
        expr = np.asarray(X).flatten()

    df["expr"] = expr

    stats = (
        df.groupby([level, meta], observed=True)
        .agg(
            mean_expr=("expr", "mean"),
            pct_pos=("expr", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
    )

    # Create pivot tables for plotting
    mean_pivot = stats.pivot(index=level, columns=meta, values="mean_expr")
    pct_pivot = stats.pivot(index=level, columns=meta, values="pct_pos")

    cell_types = mean_pivot.index.tolist()
    conditions = mean_pivot.columns.tolist()

    # Plot setup
    spacing = 1.2
    y_spacing = 0.4
    x_margin = 2
    y_margin = 1

    x_positions = [j * spacing for j in range(len(conditions))]
    y_positions = [i * y_spacing for i in range(len(cell_types))]

    fig_width = len(conditions) * spacing + x_margin
    fig_height = len(cell_types) * y_spacing + y_margin

    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color scaling
    # vmin = stats["mean_expr"].min()
    # vmax = stats["mean_expr"].max()
    vmin = 0
    vmax = 2
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    max_size = 600

    # Scatter points
    for i, ct in enumerate(cell_types):
        for j, cond in enumerate(conditions):
            mean_val = mean_pivot.loc[ct, cond]
            pct_val = pct_pivot.loc[ct, cond]

            if pd.isna(mean_val):
                continue

            size = (pct_val / 100) * max_size
            color = cmap(norm(mean_val))

            ax.scatter(
                x_positions[j],
                y_positions[i],
                s=size,
                color=color,
                edgecolors="grey",
                linewidths=0.4,
            )

    # Set axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, fontsize=12, rotation=90, ha="center")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(cell_types, fontsize=12)

    ax.set_xlim(x_positions[0] - 0.3, x_positions[-1] + 0.3)
    ax.set_ylim(y_positions[0] - 0.3, y_positions[-1] + 0.3)

    ax.set_facecolor("white")
    sns.despine(ax=ax, left=True, bottom=True)

    # Colorbar settings
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.05, aspect=12)
    cbar.set_label("Mean expression", fontsize=10)
    cbar.ax.tick_params(labelsize=12)

    # --- 7. Size legend ---
    legend_pcts = [25, 50, 75, 100]
    for pct in legend_pcts:
        ax.scatter(
            [],
            [],
            s=(pct / 100) * max_size,
            color="grey",
            edgecolors="grey",
            linewidths=0.4,
            label=f"{pct}%",
        )

    ax.legend(
        title="Fraction of cells in group (%)",
        title_fontsize=10,
        fontsize=10,
        bbox_to_anchor=(1.3, 1),
        loc="upper left",
        frameon=False,
    )

    # Save
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(
            f"{fig_dir}/{save_name}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{fig_dir}/{gene}_dotplot_{level}_{meta}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def check_genes_in_adata(adata, genes_list):
    """Check genes in list are in adata.vars.

    Args:
        adata (adata object): AnnData
        genes_list (list): List of genes to check

    Returns:
        present_genes (list): List of genes present in adata.var_names
    """
    missing_genes = [gene for gene in genes_list if gene not in adata.var_names]
    if missing_genes:
        print(
            f"Warning: The following genes are missing from adata.var_names: {missing_genes}"
        )
    else:
        print("All genes are present in adata.var_names.")

    # return list with only genes in adata.var_names
    present_genes = [gene for gene in genes_list if gene in adata.var_names]
    return present_genes


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/rejuvenair/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="dotplots")

# Set variables
level_1 = "level_1"
level_2 = "level_2"

# set subset
cell_type_subset = "Stromal cells"

# Get genes
marker_list = [
    "AXL",
    "BCL2L1",  # BCL-X
    "CTHRC1",  # CTHRC1
    "CTSB",  # Cathepsin B
    "CTSD",  # Cathepsin D
    "CTSS",  # Cathepsin S
    "CEACAM5",
    "DCN",  # Decorin
    "DKK1",
    "DLL1",
    "EGFR",  # EGF R / ErbB1
    "ENG",  # Endoglin / CD105
    "COL18A1",  # Endostatin derived
    "COL1A1",  # Collagen-I
    "COL3A1",  # Collagen-III
    "COL6A1",  # Collagen-VI
    "FGF2",  # FGF basic
    "HIF1A",  # HIF-1alpha
    "LUM",  # Lumican
    "CCL2",  # MCP-1
    "CSF1",  # M-CSF
    "CDKN1B",  # p27/Kip1
    "TP53",  # p53
    "PDGFA",  # PDGF-AA
    "SPARC",  # SPARC
    "VIM",  # Vimentin
]
dotplot_title = "remodelling_subset"
dotplot_title_safe = dotplot_title.replace(" ", "_")

# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)

# Set figure directory
folder_name = "dotplots"
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

# Subset on specific cell type
adata = adata[adata.obs["level_1_corrected"] == cell_type_subset].copy()
fig_dir = dir / fig_path / cell_type_subset.replace(" ", "_")
os.makedirs(fig_dir, exist_ok=True)
sc.settings.figdir = fig_dir  # set fig dir for plots to save to


logger.info("Data loaded successfully.")

# Check genes in list are in adata
marker_list = check_genes_in_adata(adata, marker_list)

# Subset on treatment group
logger.info("Subset on treatment arm...")
adata_treatment = adata[adata.obs["treatment_arm"].isin(["Treatment"])].copy()
adata_sham = adata[adata.obs["treatment_arm"].isin(["Sham"])].copy()

# Plot composition of treatment and sham arms across timepoints
sc.pl.dotplot(
    adata,
    marker_list,
    groupby="time_and_treatment_arm",
    title=dotplot_title,
    save=f"time_and_treatment_arm_{dotplot_title_safe}.pdf",
)

sc.pl.dotplot(
    adata_sham,
    marker_list,
    groupby="time_and_treatment_arm",
    title=dotplot_title,
    save=f"time_and_treatment_arm_{dotplot_title_safe}_SHAM.pdf",
)

sc.pl.dotplot(
    adata_treatment,
    marker_list,
    groupby="time_and_treatment_arm",
    title=dotplot_title,
    save=f"time_and_treatment_arm_{dotplot_title_safe}_TREATMENT.pdf",
)

# Plot expression
for marker in marker_list:
    # Make folder for each marker
    marker_fig_dir = fig_dir / marker
    os.makedirs(marker_fig_dir, exist_ok=True)

    # Plot full dataset
    plot_gene_dotplot(
        adata,
        marker,
        "level_2",
        "time_and_treatment_arm",
        cmap=cmap,
        fig_dir=marker_fig_dir,
        save_name=f"time_and_treatment_arm_{marker}_dotplot_level_2_timepoint",
    )

    # Plot treatment arm only
    plot_gene_dotplot(
        adata_treatment,
        marker,
        "level_2",
        "timepoint",
        cmap=cmap,
        fig_dir=marker_fig_dir,
        save_name=f"treatment_{marker}_dotplot_level_2_timepoint",
    )

    # Plot sham arm only
    plot_gene_dotplot(
        adata_sham,
        marker,
        "level_2",
        "timepoint",
        cmap=cmap,
        fig_dir=marker_fig_dir,
        save_name=f"sham_{marker}_dotplot_level_2_timepoint",
    )

    # # Plot sum per ROI
    plot_df_treatment = plot_gene_expression_per_roi(
        adata,
        new_fig_dir="_time_and_treatment_arm",
        gene=marker,
        plot_col="timepoint",
        hue_col="treatment_arm",
    )

    plot_df_treatment = plot_gene_expression_per_roi(
        adata_treatment,
        new_fig_dir="_treatment",
        gene=marker,
        plot_col="timepoint",
    )

    plot_df_sham = plot_gene_expression_per_roi(
        adata_sham,
        new_fig_dir="_sham",
        gene=marker,
        plot_col="timepoint",
    )

logger.info("All plots generated successfully.")
