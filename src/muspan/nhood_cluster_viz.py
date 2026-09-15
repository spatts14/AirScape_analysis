"""Visualize all clusters in a domain."""

import gc
from pathlib import Path

import matplotlib.pyplot as plt

import muspan as ms


def main():
    """Visualize all clusters in a domain for every domain in the directory."""
    network_type = "Delaunay"  # "Delaunay" or "proximity"
    niche_label_name = f"Neighbourhood ID {network_type}"

    # Base project path
    # base_path = Path(
    #     "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    # )
    base_path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )
    # Output directories
    outpath = base_path / "output" / "muspan" / "nb_clustering"

    domain_dir = (
        outpath
        / "domains_with_niches"
        / network_type  # Delaunay or proximity
        / "COPDvMICA_khop_1"
        / "18_clusters"
    )
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    # Create directories
    save_path = domain_dir / "plots" / "manual"
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Point to your specific folder ---
    for domain_path in domain_dir.glob("*.muspan"):
        print(f"Processing domain: {domain_path.name}...")

        domain = ms.io.load_domain(str(domain_path))

        save_path_domain = save_path / str(domain.name.replace(".muspan", ""))
        save_path_domain.mkdir(parents=True, exist_ok=True)

        # Define variables
        clusters_of_interest = [16, 17]

        # Query for cells in clusters
        label = "_".join(str(c) for c in clusters_of_interest)
        selected_clusters = ms.query.query(
            domain, ("label", niche_label_name), "in", clusters_of_interest
        )

        # Restrict to cell boundaries only
        boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

        # Combine: cells in one of the selected clusters AND in Cell boundaries
        selected_boundaries = selected_clusters & boundCells

        # Visualize
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
        plt.savefig(f"{save_path_domain}/clusters_{label}.pdf", bbox_inches="tight")
        plt.savefig(
            f"{save_path_domain}/clusters_{label}.png",
            bbox_inches="tight",
            dpi=600,
        )
        # # plt.show()
        plt.close(fig)

        # Plot cluster and cell type for each cluster
        for cluster_id in range(0, 18):
            print(f"Plotting cluster {cluster_id}...")

            selected_clusters = ms.query.query(
                domain, ("label", niche_label_name), "is", cluster_id
            )

            boundCells = ms.query.query(
                domain, ("Collection",), "is", "Cell boundaries"
            )

            selected_boundaries = selected_clusters & boundCells

            # Skip if no cells belong to this cluster in this domain
            if len(selected_boundaries) == 0:
                print(
                    f"No cells found in cluster {cluster_id} for this domain, skipping."
                )
                continue

            # Plot cluster and cell type for each cluster
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # --- Left plot: colored by neighbourhood/proximity ---
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
            # plt.show()
            plt.close(fig)

        # Plot cluster alone (final selected_boundaries from the loop above)
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
        # plt.show()
        plt.close(fig)

        # --- Free memory before loading the next domain ---
        del domain, selected_clusters, boundCells, selected_boundaries
        gc.collect()


if __name__ == "__main__":
    main()
