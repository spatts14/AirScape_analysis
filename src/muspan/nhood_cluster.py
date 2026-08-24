"""Calculate neighbourhood clusters for a given dataset and visualize the results."""

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
        "--number_of_clusters",  # need to be the same as parameter in the domain_parallel.py file
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


def build_composition_matrix(comp_df, disease_group, niche_order, celltype_order):
    """Pivot into a niche x cell_type matrix, averaged across domains, for one disease group."""
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
    if palette is None:
        palette = {}

    niche_order = sorted(comp_df["niche_id"].unique())
    celltype_order = sorted(comp_df["cell_type"].unique())

    group_1, group_2 = disease_order[0], disease_order[1]
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
            cmap="RdBu_r",
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
        cmap="RdBu_r",
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

    color_1 = palette.get(group_1, "#000000")
    color_2 = palette.get(group_2, "#000000")
    ax.set_title(
        f"Niche composition difference: {group_2} vs {group_1}\n(* = p < {alpha_level})"
    )
    # Color the axis labels/ticks to hint which side is which direction
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Niche ID")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}_difference.pdf", bbox_inches="tight")
    plt.close(fig)

    return stats_df


def main():
    """Main function to calculate neighbourhood clusters."""
    # Parse command line arguments
    number_of_clusters = parse_args(sys.argv[1:])

    # Define variables
    khop = 1  # Number of hops for neighbourhood clustering
    network_type = "proximity"  # 'Delaunay' or 'proximity'
    max_edge_distance = 30
    subset = ["IPF", "PM08"]  # COPD or IPF and PM08
    subset_safe_name = "v".join(subset)
    subset_safe_name = f"{subset_safe_name}_159removed_plots"

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
        plots_dir = plots_dir / subset_safe_name
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
        if subset is not None and not any(sub in path.stem for sub in subset):
            logger.info(
                f"Skipping {path.stem} as it does not contain any of"
                f" '{subset}' in the name"
            )
            continue
        logger.info(f"Loading {path.stem}...")
        domain = ms.io.load_domain(str(path))
        domain_list.append(domain)
    logger.info(f"Loaded {len(domain_list)} domains from {input_dir}")

    # Rewmove IPF_RBH_159 domain from the list
    domain_list = [
        domain for domain in domain_list if "IPF_RBH_159" not in str(domain.name)
    ]
    logger.info(f"After filtering, {len(domain_list)} domains remain for processing.")

    # Remove specified cell type(s) from every domain before building the network
    cell_types_to_remove = [
        "Alveolar fibroblasts (collagen high)"
    ]  # Set cell types to remove here
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
            neighbourhood_label_name=f"Neighbourhood ID {network_type}",  # Name for the neighbourhood label
            cluster_method="minibatchkmeans",  # Clustering method
            cluster_parameters={
                "n_clusters": number_of_clusters,
                "random_state": 0,
            },  # Parameters for the clustering method
            neighbourhood_enrichment_as="log-fold",  # Neighbourhood enrichment as log-fold
        )
    )

    # Create a DataFrame from the neighbourhood enrichment matrix
    df_ME_id = pd.DataFrame(
        data=neighbourhood_enrichment_matrix,
        index=unique_cluster_labels,
        columns=consistent_global_labels,
    )
    df_ME_id.index.name = f"Neighbourhood ID {network_type}"
    df_ME_id.columns.name = "Cell Type ID"

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
        # Get unique labels for the neighbourhood label
        unique_labels = np.unique(
            domain.labels[f"Neighbourhood ID {network_type}"]["labels"]
        )

        # Create a color map dict
        color_map = dict(zip(unique_labels, nb_colors[: len(unique_labels)]))

        domain.update_colors(
            color_map,
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
            / f"{network_type}_{domain_name}_{number_of_clusters}_neighbourhood_labels.pdf",
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
        / f"{network_type}_{number_of_clusters}_clusters_niche_celltype_composition.csv",
        index=False,
    )
    logger.info("Saved niche cell-type composition (per domain, per disease group).")

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
        / f"{network_type}_{number_of_clusters}_clusters_niche_composition_mannwhitney.csv",
        index=False,
    )
    logger.info("Saved niche composition comparison plots and stats.")


if __name__ == "__main__":
    main()
