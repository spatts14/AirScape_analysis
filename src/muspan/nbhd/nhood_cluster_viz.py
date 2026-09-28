"""Visualize all clusters in a domain — parallelizable via PBS job array."""

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import muspan as ms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize niche clusters for a single domain, selected by index."
    )
    parser.add_argument(
        "--domain_index",
        type=int,
        required=True,
        help="Index into the sorted list of domain files to process (0-based).",
    )
    return parser.parse_args()


def main():
    """Visualize all clusters for a single domain, selected via --domain_index."""
    args = parse_args()

    network_type = "proximity"  # "Delaunay" or "proximity"
    niche_label_name = f"Neighbourhood ID {network_type}"
    dir_name = "PM08vIPF_khop_1"
    clusters_of_interest = [5, 9, 17, 10]
    cluster = 18

    # Base project path
    base_path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )

    outpath = base_path / "output" / "muspan" / "nb_clustering"

    domain_dir = (
        outpath
        / "domains_with_niches"
        / network_type
        / dir_name
        / f"{cluster}_clusters"
    )
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    save_path = domain_dir / "plots" / "manual"
    save_path.mkdir(parents=True, exist_ok=True)

    # Sorted so every array task sees the same, deterministic ordering —
    # critical for --domain_index to reliably map to the same file every run.
    domain_paths = sorted(domain_dir.glob("*.muspan"))

    if not domain_paths:
        raise FileNotFoundError(f"No .muspan files found in {domain_dir}")

    if args.domain_index >= len(domain_paths):
        raise IndexError(
            f"--domain_index {args.domain_index} out of range: "
            f"only {len(domain_paths)} domain files found."
        )

    domain_path = domain_paths[args.domain_index]
    print(
        f"Processing domain {args.domain_index}/{len(domain_paths) - 1}: {domain_path.name}..."
    )

    domain = ms.io.load_domain(str(domain_path))
    boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    save_path_domain = save_path / str(domain.name.replace(".muspan", ""))
    save_path_domain.mkdir(parents=True, exist_ok=True)

    # Update color
    domain.update_colors(
        {"17": "#8B7CB3"}, colors_to_update="labels", label_name=niche_label_name
    )

    # --- Plot combined clusters of interest, before the per-cluster loop
    # below reassigns selected_boundaries ---
    label = "_".join(str(c) for c in clusters_of_interest)

    selected_clusters = ms.query.query(
        domain, ("label", niche_label_name), "in", clusters_of_interest
    )
    selected_boundaries = selected_clusters & boundCells

    print(f"Visualizing clusters {label}...")
    fig, ax = plt.subplots(figsize=(8, 6))

    ms.visualise.visualise(
        domain,
        objects_to_plot=boundCells,
        add_cbar=False,
        shape_kwargs={
            "alpha": 0.5,
            "linewidth": 0.005,
            "edgecolor": "#00000000",
            "color": "#848484",
        },
        ax=ax,
    )

    ms.visualise.visualise(
        domain,
        color_by=niche_label_name,
        objects_to_plot=selected_boundaries,
        shape_kwargs=dict(alpha=1, linewidth=0.001, edgecolor="#00000000"),
        ax=ax,
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
    plt.savefig(
        f"{save_path_domain}/clusters_{label}.pdf", bbox_inches="tight", dpi=300
    )
    plt.savefig(
        f"{save_path_domain}/clusters_{label}.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close(fig)

    # --- Plot cluster and cell type for each individual cluster ---
    for cluster_id in range(0, 18):
        print(f"Plotting cluster {cluster_id}...")

        niche_labels = np.asarray(domain.labels[niche_label_name]["labels"])
        n_cells_in_cluster = int(np.sum(niche_labels == cluster_id))
        print(f"Number of cells in cluster {cluster_id}: {n_cells_in_cluster}")

        if n_cells_in_cluster == 0:
            print(f"No cells found in cluster {cluster_id} for this domain, skipping.")
            continue

        selected_clusters = ms.query.query(
            domain, ("label", niche_label_name), "is", cluster_id
        )
        selected_boundaries = selected_clusters & boundCells

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Left plot: colored by neighbourhood ---
        ax = axes[0]
        print("Plotting left plot...")
        ms.visualise.visualise(
            domain,
            objects_to_plot=boundCells,
            add_cbar=False,
            shape_kwargs={
                "alpha": 0.5,
                "linewidth": 0.005,
                "edgecolor": "#00000000",
                "color": "#848484",
            },
            ax=ax,
        )

        ms.visualise.visualise(
            domain,
            color_by=niche_label_name,
            objects_to_plot=selected_boundaries,
            shape_kwargs=dict(alpha=1, linewidth=0.001, edgecolor="#00000000"),
            ax=ax,
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

        # --- Right plot: colored by cell type ---
        ax = axes[1]
        print("Plotting right plot...")
        ms.visualise.visualise(
            domain,
            objects_to_plot=boundCells,
            add_cbar=False,
            shape_kwargs={
                "alpha": 0.5,
                "linewidth": 0.005,
                "edgecolor": "#00000000",
                "color": "#848484",
            },
            ax=ax,
        )

        ms.visualise.visualise(
            domain,
            color_by="Cell Type",
            objects_to_plot=selected_boundaries,
            shape_kwargs=dict(alpha=1, linewidth=0.001, edgecolor="#00000000"),
            ax=ax,
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

        plt.tight_layout()
        print("Saving combined plot...")
        plt.savefig(
            f"{save_path_domain}/combined_plot_{cluster_id}.png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.close(fig)

        # --- "Cluster alone" plot for THIS cluster ---
        fig, ax = plt.subplots(figsize=(8, 6))

        ms.visualise.visualise(
            domain,
            objects_to_plot=boundCells,
            add_cbar=False,
            shape_kwargs={
                "alpha": 0.5,
                "linewidth": 0.005,
                "edgecolor": "#00000000",
                "color": "#848484",
            },
            ax=ax,
        )

        ms.visualise.visualise(
            domain,
            color_by=niche_label_name,
            objects_to_plot=selected_boundaries,
            shape_kwargs=dict(alpha=1, linewidth=0.001, edgecolor="#00000000"),
            ax=ax,
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
        plt.tight_layout()
        plt.savefig(
            f"{save_path_domain}/cluster_only_{cluster_id}.png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.close(fig)

    # --- Free memory ---
    del domain, selected_clusters, boundCells, selected_boundaries
    gc.collect()

    print(f"Finished processing {domain_path.name}.")


if __name__ == "__main__":
    main()
