"""Calculate neighborhood clusters for a given dataset and visualize the results."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.airspace_colors import diagnosis_palette
from utils.setup_logger import setup_logger


def remove_cell_types(domain, cell_types_to_remove, label_name="Cell Type"):
    """Remove all objects of the given cell type(s) from a domain, in place.

    Uses muspan's query interface to find objects matching the label(s),
    then deletes them from the domain so downstream network construction
    (e.g. cluster_neighbourhoods) only sees the remaining cells.
    """
    if not cell_types_to_remove:
        return domain

    query_result = ms.query.query(
        domain, ("label", label_name), "in", cell_types_to_remove
    )
    domain.delete_objects(query_result)
    return domain


def parse_args(args):
    """Parse command line arguments for the script.

    Args:
        args: List of command line arguments (excluding the script name).

    Returns:
        Tuple containing the ROI name and the path to the Xenium data directory.
    """
    parser = argparse.ArgumentParser(description="Map cell types to a domain")

    parser.add_argument(  # class for parameters
        "-c",  # shortcut
        "--number_of_clusters",  # need to be same as parameter in domain_parallel.py
        help="Number of clusters to pass [required]",
        type=int,
        dest="number_of_clusters",  # how you will call this variable in the code
        action="store",  # store the value provided in the command line
        required=True,
    )

    results = parser.parse_args(args)
    return results.number_of_clusters


def get_disease_group(domain_name, subset):
    """Map a domain name to its disease group based on substring match."""
    for group in subset:
        if group in domain_name:
            return group
    return "Unknown"


def compute_niche_celltype_composition(
    domain_list, network_type, subset, label_name="Cell Type"
):
    """Per-domain cell-type composition of each niche.

    For each domain, cross-tabulates niche assignment against cell type,
    giving the fraction of cells within each niche that belong to each
    cell type — tagged by domain and disease group so it can be split
    or averaged by ROI or by diagnosis afterward.
    """
    niche_label_name = f"Neighbourhood ID {network_type}"
    records = []

    for domain in domain_list:
        niche_labels = np.asarray(domain.labels[niche_label_name]["labels"])
        cell_types = np.asarray(domain.labels[label_name]["labels"])
        disease_group = get_disease_group(str(domain.name), subset)

        df = pd.DataFrame({"niche_id": niche_labels, "cell_type": cell_types})

        for niche_id, niche_group in df.groupby("niche_id"):
            n_total = len(niche_group)
            counts = niche_group["cell_type"].value_counts()
            for cell_type, count in counts.items():
                records.append(
                    {
                        "domain": str(domain.name),
                        "disease_group": disease_group,
                        "niche_id": str(niche_id),
                        "cell_type": cell_type,
                        "n_cells": count,
                        "proportion_within_niche": count / n_total,
                    }
                )

    return pd.DataFrame.from_records(records)


def compute_niche_proportions_by_domain(domain_list, network_type, subset):
    """Per-domain (per-ROI) niche proportions: what % of each ROI's cells fall.

    into each niche. Total cells in the ROI is the denominator; cells in the
    niche is the numerator.

    Returns a long dataframe with columns: domain, disease_group, niche_id,
    n_cells, total_cells, pct.
    """
    niche_label_name = f"Neighbourhood ID {network_type}"
    records = []

    for domain in domain_list:
        niche_labels = np.asarray(domain.labels[niche_label_name]["labels"])
        disease_group = get_disease_group(str(domain.name), subset)
        total_cells = len(niche_labels)

        unique_niches, counts = np.unique(niche_labels, return_counts=True)

        for niche_id, count in zip(unique_niches, counts):
            records.append(
                {
                    "domain": str(domain.name),
                    "disease_group": disease_group,
                    "niche_id": str(niche_id),
                    "n_cells": count,
                    "total_cells": total_cells,
                    "pct": 100 * count / total_cells,
                }
            )

    return pd.DataFrame.from_records(records)


def pivot_niche_pct_wide(prop_df, niche_order):
    """Pivot the long per-domain niche-proportion dataframe into a wide table.

    rows = domain (ROI), columns = niche_id, values = pct. Missing niches
    (i.e. a niche absent from a given ROI) are filled with 0, not dropped —
    this matters for correct averaging later, since an absent niche should
    count as 0% for that ROI rather than being excluded from any mean.

    A 'disease_group' column is attached per domain for downstream grouping.
    """
    wide = prop_df.pivot(index="domain", columns="niche_id", values="pct").fillna(0)
    wide = wide.reindex(columns=niche_order, fill_value=0)

    domain_to_group = prop_df.drop_duplicates("domain").set_index("domain")[
        "disease_group"
    ]
    wide["disease_group"] = domain_to_group

    return wide


def build_composition_matrix(comp_df, disease_group, niche_order, celltype_order):
    """Pivot into a niche x cell_type matrix averaged across domains for one disease."""
    sub = comp_df[comp_df["disease_group"] == disease_group]
    pivot = (
        sub.groupby(["niche_id", "cell_type"])["proportion_within_niche"]
        .mean()
        .unstack(fill_value=0)
    )
    return pivot.reindex(index=niche_order, columns=celltype_order, fill_value=0)


def compute_composition_diff_stats(comp_df, disease_order, alpha_level=0.05):
    """Per niche x cell_type, run Mann-Whitney across domains between disease groups.

    Returns a long dataframe with the mean proportion in each group, the
    difference (group_2 - group_1), and the p-value, one row per (niche, cell_type).
    """
    group_1, group_2 = disease_order[0], disease_order[1]
    records = []

    for (niche_id, cell_type), grp in comp_df.groupby(["niche_id", "cell_type"]):
        vals_1 = grp.loc[
            grp["disease_group"] == group_1, "proportion_within_niche"
        ].to_numpy()
        vals_2 = grp.loc[
            grp["disease_group"] == group_2, "proportion_within_niche"
        ].to_numpy()

        if len(vals_1) < 1 or len(vals_2) < 1:
            continue

        mean_1 = vals_1.mean()
        mean_2 = vals_2.mean()

        if (
            len(vals_1) >= 1
            and len(vals_2) >= 1
            and (len(vals_1) > 1 or len(vals_2) > 1)
        ):
            try:
                stat, p_value = mannwhitneyu(vals_1, vals_2)
            except ValueError:
                # e.g. all values identical
                stat, p_value = np.nan, np.nan
        else:
            stat, p_value = np.nan, np.nan

        records.append(
            {
                "niche_id": niche_id,
                "cell_type": cell_type,
                f"mean_{group_1}": mean_1,
                f"mean_{group_2}": mean_2,
                "diff": mean_2 - mean_1,
                "statistic": stat,
                "p_value": p_value,
                "significant": (p_value < alpha_level) if pd.notna(p_value) else False,
            }
        )

    return pd.DataFrame.from_records(records)


def plot_composition_comparison(
    comp_df, disease_order, out_path_prefix, palette=None, alpha_level=0.05
):
    """Plot side-by-side niche x cell-type composition heatmaps for two disease groups,
    plus a diverging difference heatmap with significance markers.

    Saves two files: '{prefix}_side_by_side.pdf', '{prefix}_difference.pdf',
    and returns the underlying stats dataframe.
    """  # noqa: D205
    color_0_1 = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    if palette is None:
        palette = {}

    niche_order = sorted(comp_df["niche_id"].unique())
    celltype_order = sorted(comp_df["cell_type"].unique())

    group_1, group_2 = disease_order[1], disease_order[0]
    mat_1 = build_composition_matrix(comp_df, group_1, niche_order, celltype_order)
    mat_2 = build_composition_matrix(comp_df, group_2, niche_order, celltype_order)

    vmax = max(mat_1.values.max(), mat_2.values.max())

    # --- Side-by-side heatmaps ---
    fig, axes = plt.subplots(
        1, 2, figsize=(0.5 * len(celltype_order) * 2 + 4, 0.5 * len(niche_order) + 3)
    )
    for ax, mat, title in zip(axes, [mat_1, mat_2], [group_1, group_2]):
        sns.heatmap(
            mat,
            ax=ax,
            cmap=color_0_1,  # scale 0 to 1
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Proportion within niche"},
        )
        title_color = palette.get(title, "#000000")
        ax.set_title(title, color=title_color, fontweight="bold")
        ax.set_xlabel("Cell type")
        ax.set_ylabel("Niche ID")
        ax.tick_params(axis="x", rotation=90)

        # Colored border framing the panel, tying it to its disease group
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(title_color)
            spine.set_linewidth(2.5)

    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Difference heatmap with stats ---
    stats_df = compute_composition_diff_stats(
        comp_df, disease_order, alpha_level=alpha_level
    )
    diff_mat = stats_df.pivot(index="niche_id", columns="cell_type", values="diff")
    diff_mat = diff_mat.reindex(index=niche_order, columns=celltype_order, fill_value=0)

    sig_mat = stats_df.pivot(
        index="niche_id", columns="cell_type", values="significant"
    )
    sig_mat = sig_mat.reindex(
        index=niche_order, columns=celltype_order, fill_value=False
    )

    diff_abs_max = np.nanmax(np.abs(diff_mat.values)) if diff_mat.size else 1

    fig, ax = plt.subplots(
        figsize=(0.5 * len(celltype_order) + 4, 0.5 * len(niche_order) + 3)
    )
    sns.heatmap(
        diff_mat,
        ax=ax,
        cmap=cmap,  # scale -1 to 1
        center=0,
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Δ proportion ({group_2} - {group_1})"},
    )

    for i, niche_id in enumerate(niche_order):
        for j, cell_type in enumerate(celltype_order):
            if sig_mat.loc[niche_id, cell_type]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "*",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.set_title(
        f"Niche composition difference: {group_2} vs {group_1}\n(* = p < {alpha_level})"
    )
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Niche ID")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}_difference.pdf", bbox_inches="tight")
    plt.close(fig)

    return stats_df


def plot_niche_pct_stacked_bar(
    pivot_df,
    niche_order,
    niche_color_map,
    out_path,
    xlabel,
    title=None,
    figsize=None,
    row_order=None,
):
    """Stacked bar plot of niche percentage composition.

    One bar per row of pivot_df (e.g. one bar per ROI, or one bar per disease group),
    colored by each niche's assigned color.

    Args:
        pivot_df : pd.DataFrame
            DataFrame with rows = ROI or disease group.
            Columns = niche_id, values = pct.
        niche_order : list
            List of niche IDs in the order they should appear in the stacked bars.
        niche_color_map : dict
            Mapping of niche IDs to colors (hex or RGB).
        out_path : str or Path
            Path to save the figure.
        xlabel : str
            Label for the x-axis.
        title : str, optional
            Title for the plot.
        figsize : tuple, optional
            Size of the figure (width, height). If None, a default size is calculated.
        row_order : list, optional
            Explicit order for the bars (index of pivot_df), e.g. ["MICA", "COPD"].
            Rows not listed are appended afterward in their existing order, rather
            than being dropped.

    """
    if row_order is not None:
        present = [r for r in row_order if r in pivot_df.index]
        remaining = [r for r in pivot_df.index if r not in present]
        pivot_df = pivot_df.reindex(present + remaining)

    if figsize is None:
        figsize = (max(6, 0.4 * len(pivot_df)), 5)

    colors = [niche_color_map.get(n, "#888888") for n in niche_order]

    fig, ax = plt.subplots(figsize=figsize)
    pivot_df[niche_order].plot(kind="bar", stacked=True, color=colors, ax=ax)

    ax.grid(False)
    ax.set_ylabel("Percentage of cells (%)")
    ax.set_xlabel(xlabel)
    ax.legend(title="Niche", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=90)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def should_load_domain(stem):
    """True if this domain file should be loaded.

    Loads MICA sample (no timepoint restriction), or a COPD sample at the V1
    timepoint. Everything else (IPF, PM08, COPD V2/V3, etc.) is skipped.
    """
    # MICA and COPD
    if "MICA" in stem:
        return True
    if "COPD" in stem:
        return "_V1_" in stem
    return False


def pivot_niche_count_wide(prop_df, niche_order):
    """Pivot the long per-domain niche-proportion dataframe into a wide table
    of raw cell counts: rows = domain (ROI), columns = niche_id, values = n_cells.

    Missing niches (absent from a given ROI) are filled with 0, not dropped —
    an absent niche is a real 0, not missing data.
    """
    wide = prop_df.pivot(index="domain", columns="niche_id", values="n_cells").fillna(0)
    wide = wide.reindex(columns=niche_order, fill_value=0)
    wide[niche_order] = wide[niche_order].astype(int)

    domain_to_group = prop_df.drop_duplicates("domain").set_index("domain")[
        "disease_group"
    ]
    wide["disease_group"] = domain_to_group

    return wide


def plot_niche_count_heatmap(
    wide_df,
    niche_order,
    out_path,
    palette=None,
    row_order=None,
    title=None,
    figsize=None,
    cmap="viridis",
):
    """Heatmap of raw cell counts: rows = domain (ROI), columns = niche.

    Domains are grouped by disease group (per row_order) then sorted by
    name; each y-tick label is colored by its disease group's palette
    color so the grouping is visible without a separate legend.
    """
    if palette is None:
        palette = {}

    df = wide_df.copy()

    if row_order is not None:
        group_rank = {g: i for i, g in enumerate(row_order)}
        sort_df = df.reset_index()
        sort_df["_group_rank"] = (
            sort_df["disease_group"].map(group_rank).fillna(len(row_order))
        )
        sort_df = sort_df.sort_values(["_group_rank", "domain"]).drop(
            columns="_group_rank"
        )
        df = sort_df.set_index("domain")
    else:
        df = df.sort_index()

    disease_groups = df["disease_group"]
    count_matrix = df[niche_order]

    if figsize is None:
        figsize = (max(6, 0.5 * len(niche_order) + 2), max(4, 0.3 * len(df)))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        count_matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt="d",
        cbar_kws={"label": "Number of cells"},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Niche ID")
    ax.set_ylabel("Domain (ROI)")
    ax.tick_params(axis="x", rotation=90)
    if title:
        ax.set_title(title)

    for tick_label, domain_name in zip(ax.get_yticklabels(), count_matrix.index):
        group = disease_groups.loc[domain_name]
        tick_label.set_color(palette.get(group, "#000000"))
        tick_label.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """Main function to calculate neighbourhood clusters."""
    # Parse command line arguments
    number_of_clusters = parse_args(sys.argv[1:])

    # Define variables
    khop = 1  # Number of hops for neighbourhood clustering
    network_type = "Delaunay"  # 'Delaunay' or 'proximity'
    max_edge_distance = 30
    subset = ["MICA", "COPD"]  # COPD or IPF and PM08
    subset_safe_name = "v".join(subset)
    subset_safe_name = f"{subset_safe_name}"

    # Base project path
    paths = [
        Path(
            "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
        ),
        Path(
            "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
        ),
    ]

    base_path = next((p for p in paths if p.exists()), None)

    if base_path is None:
        raise FileNotFoundError("None of the candidate base paths exist.")

    print(f"Using base path: {base_path}")

    # Input
    input_dir = base_path / "output" / "muspan" / "domains"

    # Output directories
    outpath = base_path / "output" / "muspan" / "nb_clustering"
    data_dir = outpath / "data" / network_type
    plots_dir = outpath / "plots" / network_type

    # Create directories
    for path in [outpath, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logs_dir = Path(base_path) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        log_dir=logs_dir, log_name=f"nhood_cluster_{number_of_clusters}"
    )

    # Define a color palette for the neighbourhood labels
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    nb_colors = [
        "#5B8FA8",  # dusty blue
        "#4E7D5B",  # muted forest green
        "#8B7CB3",  # dusty lavender
        "#A05A4A",  # muted brick red
        "#E8B4A0",  # dusty peach
        "#7E9E6E",  # sage green
        "#B5C99A",  # soft moss
        "#5A7A9A",  # slate blue
        "#B8A06E",  # muted gold
        "#783129",  # deep rust
        "#E99547",  # warm amber
        "#7A9E9A",  # dusty teal
        "#C4956A",  # muted terracotta
        "#6B5B95",  # dusty plum
        "#C97B63",  # clay orange
        "#4A7C6F",  # deep teal green
        "#D4A5A5",  # dusty rose
        "#8A9B6E",  # olive green
        "#B06C8F",  # muted magenta
        "#5D8A66",  # muted emerald
    ]

    # If subset is specified, create a subdirectory for plots
    if subset is not None:
        plots_dir = plots_dir / f"{subset_safe_name}"  # final dir name
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Subset specified: {subset}. Plots will be saved to {plots_dir}")

    # Load the domain from file
    # Add domain to list
    domain_list = []

    # domains stored in directory
    logger.info(f"Loading domains from {input_dir}...")
    for path in input_dir.glob("*.muspan"):
        if not path.is_file():
            logger.warning(f"Skipping {path.stem} as it is not a file.")
            continue
        if not should_load_domain(path.stem):
            logger.info(f"Skipping {path.stem} (not MICA, or not a COPD V1 sample).")
            continue
        logger.info(f"Loading {path.stem}...")
        domain = ms.io.load_domain(str(path))
        domain_list.append(domain)
    logger.info(f"Loaded {len(domain_list)} domains from {input_dir}")

    # Remove specified cell type(s) from every domain before building the network
    cell_types_to_remove = [
        "Alveolar fibroblasts",
        "Alveolar fibroblasts (collagen high)",
        "AT1 cells",
        "AT2 cells",
        "Lipid-associated macrophages",
        "Airway/Alveolar macrophages",
        "Proliferating AT2 cells",
    ]

    if cell_types_to_remove:
        logger.info(f"Removing cell types {cell_types_to_remove} from all domains...")
        for domain in domain_list:
            n_before = domain.n_objects if hasattr(domain, "n_objects") else None
            remove_cell_types(domain, cell_types_to_remove, label_name="Cell Type")
            n_after = domain.n_objects if hasattr(domain, "n_objects") else None
            if n_before is not None and n_after is not None:
                logger.info(
                    f"{domain.name}: removed {n_before - n_after} cells "
                    f"({n_before} -> {n_after})"
                )

    # Make cluster number of clusters
    plots_dir_cluster = plots_dir / f"{number_of_clusters}_clusters"
    plots_dir_cluster.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Processing neighbourhood clustering with {number_of_clusters} clusters."
        f"Plots will be saved to {plots_dir_cluster}"
    )

    # Perform neighbourhood clustering on the dataset using KNN and minibatchkmeans
    logger.info(
        f"Performing neighbourhood clustering with {network_type} network and"
        f"{number_of_clusters} clusters..."
    )
    neighbourhood_enrichment_matrix, consistent_global_labels, unique_cluster_labels = (
        ms.networks.cluster_neighbourhoods(
            domain_list,  # The domain dataset
            label_name="Cell Type",  # The label to use for clustering
            network_kwargs=dict(
                network_type=network_type,
                max_edge_distance=max_edge_distance,
                min_edge_distance=0,
            ),  # The network parameters
            k_hops=khop,  # The number of hops to consider for the neighbourhood
            neighbourhood_label_name=f"Neighbourhood ID {network_type}",
            cluster_method="minibatchkmeans",  # Clustering method
            cluster_parameters={
                "n_clusters": number_of_clusters,
                "random_state": 0,
            },  # Parameters for the clustering method
            neighbourhood_enrichment_as="log-fold",  # Neighbourhood enrichment as LF
        )
    )

    # Build ONE global niche -> color mapping, used consistently everywhere
    # (per-domain visualization, ROI stacked bar, condition stacked bar) so a
    # given niche always has the same color regardless of which domains it
    # happens to appear in.
    niche_order = [str(label) for label in unique_cluster_labels]
    niche_color_map = dict(zip(niche_order, nb_colors[: len(niche_order)]))
    domain_color_map = dict(
        zip(unique_cluster_labels, nb_colors[: len(unique_cluster_labels)])
    )

    # Create a DataFrame from the neighbourhood enrichment matrix
    df_ME_id = pd.DataFrame(
        data=neighbourhood_enrichment_matrix,
        index=unique_cluster_labels,
        columns=consistent_global_labels,
    )
    df_ME_id.index.name = f"Neighbourhood ID {network_type}"
    df_ME_id.columns.name = "Cell Type ID"

    # Safety net: consistent_global_labels can retain removed cell types even after
    # domain.delete_objects() (muspan's internal label vocabulary doesn't always get
    # rebuilt after deletion) — drop them explicitly so the heatmap matches reality.
    cols_to_drop = [c for c in df_ME_id.columns if c in cell_types_to_remove]
    if cols_to_drop:
        logger.warning(
            f"Removed cell types still present in consistent_global_labels: "
            f"{cols_to_drop} — dropping them from the heatmap and enrichment matrix."
        )
        df_ME_id = df_ME_id.drop(columns=cols_to_drop)
        consistent_global_labels = [
            c for c in consistent_global_labels if c not in cell_types_to_remove
        ]

    # Filter out sentinel values before computing range
    logger.info(
        "Filtering out sentinel values from neighbourhood"
        "enrichment matrix for visualization"
    )
    finite_vals = df_ME_id.values[
        np.isfinite(df_ME_id.values) & (np.abs(df_ME_id.values) < 1e300)
    ]
    vmin = np.floor(finite_vals.min())
    vmax = np.ceil(finite_vals.max())
    df_plot = df_ME_id.clip(lower=vmin, upper=vmax)
    logger.info(
        f"Neighbourhood enrichment matrix value range before filtering:"
        f" min={finite_vals.min()}, max={finite_vals.max()}"
    )

    # plotting vmax and vimin for the clustermap
    plot_vmin = -5
    plot_vmax = 5

    logger.info(
        f"Neighbourhood enrichment matrix value range after filtering:"
        f" min={vmin}, max={vmax}"
    )

    # Make sure the data output directory exists
    data_output_dir = data_dir / subset_safe_name
    data_output_dir.mkdir(parents=True, exist_ok=True)

    df_plot.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_neighbourhood_enrichment.csv"
    )

    # Visualize the neighbourhood enrichment matrix using a clustermap
    logger.info("Visualizing the neighbourhood enrichment matrix using a clustermap...")
    sns.clustermap(
        df_plot,
        xticklabels=consistent_global_labels,
        yticklabels=unique_cluster_labels,
        figsize=(10, 10),
        cmap=cmap,
        dendrogram_ratio=(0.05, 0.3),
        col_cluster=True,
        row_cluster=True,
        square=True,
        linewidths=0.5,
        linecolor="black",
        cbar_kws=dict(
            use_gridspec=False,
            location="top",
            label="Neighbourhood enrichment (log-fold)",
            ticks=[plot_vmin, 0, plot_vmax],
        ),
        cbar_pos=(0.12, 1.05, 0.72, 0.06),
        vmin=plot_vmin,
        vmax=plot_vmax,
        tree_kws={"linewidths": 1, "color": "black"},
    )
    plt.suptitle(
        f"{network_type.capitalize()} Neighbourhood Enrichment Clustering",
        fontsize=14,
        y=1.3,
    )
    plt.savefig(
        plots_dir_cluster
        / f"{network_type}_{number_of_clusters}_clusters_neighbourhood_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close()

    for domain in domain_list:
        # Use the single global niche color map (built above from
        # unique_cluster_labels) instead of re-deriving colors from just
        # this domain's own unique labels, so colors are identical across
        # every domain and every downstream plot.
        domain.update_colors(
            domain_color_map,
            colors_to_update="labels",
            label_name=f"Neighbourhood ID {network_type}",
        )

        # Set domain name
        domain_name = str(domain.name)

        # Get cell centroids for plotting
        qCells = ms.query.query(domain, ("Collection",), "is", "Cell centroids")

        # Visualize the domain with neighbourhood labels
        logger.info(f"Visualizing domain {domain_name} with neighbourhood labels...")
        ms.visualise.visualise(
            domain,
            color_by=f"Neighbourhood ID {network_type}",
            marker_size=0.8,
            objects_to_plot=qCells,
            add_scalebar=True,
            scalebar_kwargs={
                "size": 500,
                "label": "500µm",
                "loc": "lower right",
                "pad": 0.1,
                "color": "black",
                "frameon": False,
                "size_vertical": 2,
            },
        )
        plt.suptitle(
            f"Domain Visualization with Neighbourhood Labels for {domain_name}", y=1.2
        )
        plt.savefig(
            plots_dir_cluster
            / f"{network_type}_{domain_name}_{number_of_clusters}.pdf",
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Finished processing domain {domain_name} with"
            f" {number_of_clusters} clusters."
        )

    # Compute per-domain, per-disease-group niche cell-type composition
    logger.info("Computing per-domain niche cell-type composition...")
    comp_df = compute_niche_celltype_composition(domain_list, network_type, subset)
    comp_df.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_celltype_composition.csv",  # Noqa: E501
        index=False,
    )
    logger.info("Saved niche cell-type composition (per domain, per disease group).")

    # --- Niche proportion stacked bar plots: by ROI and by condition ---
    logger.info("Computing per-domain (per-ROI) niche proportions...")
    niche_prop_df = compute_niche_proportions_by_domain(
        domain_list, network_type, subset
    )
    niche_prop_df.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_proportions_by_roi.csv",
        index=False,
    )

    niche_prop_wide = pivot_niche_pct_wide(niche_prop_df, niche_order)

    logger.info("Plotting niche proportion stacked bar by ROI...")
    plot_niche_pct_stacked_bar(
        niche_prop_wide,
        niche_order=niche_order,
        niche_color_map=niche_color_map,
        out_path=plots_dir_cluster
        / f"{network_type}_{number_of_clusters}_clusters_niche_pct_by_roi.pdf",
        xlabel="ROI",
        title="Niche composition by ROI (% of cells)",
    )

    logger.info(
        "Plotting niche proportion stacked bar by condition "
        "(averaged across ROIs within each condition)..."
    )
    # Average each ROI's niche % within its disease group — every ROI
    # contributes equally regardless of its cell count.
    niche_pct_by_condition = niche_prop_wide.groupby("disease_group")[
        niche_order
    ].mean()
    niche_pct_by_condition.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_pct_by_condition.csv"
    )

    plot_niche_pct_stacked_bar(
        niche_pct_by_condition,
        niche_order=niche_order,
        niche_color_map=niche_color_map,
        out_path=plots_dir_cluster
        / f"{network_type}_{number_of_clusters}_clusters_niche_pct_by_condition.pdf",
        xlabel="Condition",
        title="Niche composition by condition (mean %, averaged across ROIs)",
        figsize=(5, 5),
        row_order=subset,  # e.g. ["MICA", "COPD"] — controls bar order explicitly
    )

    # Compare niche composition between disease groups
    logger.info(f"Comparing niche composition between {subset[0]} and {subset[1]}...")
    comp_stats_df = plot_composition_comparison(
        comp_df,
        disease_order=subset,
        out_path_prefix=str(
            plots_dir_cluster
            / f"{network_type}_{number_of_clusters}_clusters_niche_composition"
        ),
        palette=diagnosis_palette,
    )
    comp_stats_df.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_composition_mannwhitney.csv",  # Noqa: E501
        index=False,
    )
    logger.info("Saved niche composition comparison plots and stats.")

    # --- Niche proportions by domain (ROI) ---
    niche_prop_df = compute_niche_proportions_by_domain(
        domain_list, network_type, subset
    )
    niche_prop_df.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_proportions_by_roi.csv",
        index=False,
    )

    # --- Niche cell-count heatmap: domain (row) x niche (column) ---
    niche_count_wide = pivot_niche_count_wide(niche_prop_df, niche_order)
    niche_count_wide.to_csv(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_counts_by_roi.csv"
    )
    niche_count_wide.to_excel(
        data_output_dir
        / f"{network_type}_{number_of_clusters}_clusters_niche_counts_by_roi.xlsx",
        sheet_name="niche_counts_by_roi",
    )

    logger.info("Plotting niche cell-count heatmap (domain x niche)...")
    plot_niche_count_heatmap(
        niche_count_wide,
        niche_order=niche_order,
        out_path=plots_dir_cluster
        / f"{network_type}_{number_of_clusters}_clusters_niche_counts_heatmap.pdf",
        palette=diagnosis_palette,
        row_order=subset,
        title="Number of cells per niche, by domain (ROI)",
    )


if __name__ == "__main__":
    main()
