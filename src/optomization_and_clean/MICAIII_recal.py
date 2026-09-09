"""Re-calculate MuSpAn stats the MICA III domain."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import muspan as ms

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    """Load MICA III domains and recalculate cell types, networks, and APT."""
    base_dir = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )

    out_dir = base_dir / "output" / "muspan"
    domains_dir = out_dir / "domains"
    figs_dir = out_dir / "figures"
    ADJ_perm_dir = out_dir / "adjacency_permutation_test_results"

    domain_list = [
        "MICA_III_311",
        "MICA_III_315",
        "MICA_III_319",
        "MICA_III_325",
        "MICA_III_337",
        "MICA_III_379",
    ]
    for roi in domain_list:
        saved_domain_path = domains_dir / f"{roi}_muspan_domain.muspan"
        domain = ms.io.load_domain(str(saved_domain_path))
        print(f"Reloaded domain: {domain}")

        # Update domain name
        domain.name = roi

        # Create output directories for this ROI
        roi_dir = figs_dir / roi
        roi_dir.mkdir(parents=True, exist_ok=True)

        # Query to isolate Cell centroids for visualization
        print("Querying domain to isolate cell centroids for visualization")
        boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")
        centCells = ms.query.query(domain, ("Collection",), "is", "Cell centroids")

        # Visualize Cell Types
        print(
            f"Visualize the MuSpAn domain for {roi} with cell types and cell boundaries"
        )
        ms.visualise.visualise(
            domain,
            color_by=("label", "Cell Type"),
            objects_to_plot=boundCells,
            shape_kwargs=dict(alpha=1, linewidth=0.01, edgecolor="#00000000"),
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
        plt.savefig(roi_dir / f"{roi}_cell_types_boundaries.png")
        plt.savefig(roi_dir / f"{roi}_cell_types_boundaries.pdf")

        print(
            f"Visualize the MuSpAn domain for {roi} with cell types and cell centroids"
        )
        ms.visualise.visualise(
            domain,
            color_by=("label", "Cell Type"),
            objects_to_plot=centCells,
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
            marker_size=0.5,
        )
        plt.savefig(roi_dir / f"{roi}_cell_types_centroids.png")
        plt.savefig(roi_dir / f"{roi}_cell_types_centroids.pdf")

        print("Generating spatial networks for the domain...")
        # Generate spatial networks
        # Delaunay network
        ms.networks.generate_network(
            domain,
            network_name="Delaunay CC",
            network_type="Delaunay",
            objects_as_nodes=("collection", "Cell centroids"),
        )

        ms.networks.generate_network(
            domain,
            network_name="Delaunay CC filtered",
            network_type="Delaunay",
            objects_as_nodes=("collection", "Cell centroids"),
            min_edge_distance=0,
            max_edge_distance=30,
        )

        # Proximity network with 30μm max distance
        ms.networks.generate_network(
            domain,
            network_name="Proximity_30um",
            network_type="Proximity",
            objects_as_nodes=("collection", "Cell centroids"),
            max_edge_distance=30,
            min_edge_distance=0,
        )

        print(f"Visualize the networks for {roi}")
        # Plot the original Delaunay network
        ms.visualise.visualise_network(
            domain,
            network_name="Delaunay CC",
            edge_weight_name=None,
            edge_width=0.2,
            edge_cmap="#060606",
            add_cbar=False,
            visualise_kwargs=dict(
                objects_to_plot=("collection", "Cell centroids"),
                marker_size=0.5,
                add_cbar=True,
                color_by=("label", "Cell Type"),
                scatter_kwargs=dict(  # ← linewidths/edgecolors go HERE
                    edgecolors="none",
                ),
            ),
        )

        plt.savefig(roi_dir / f"{roi}_delaunay_cc.png")
        plt.savefig(roi_dir / f"{roi}_delaunay_cc.pdf")

        # Plot the filtered Delaunay network
        ms.visualise.visualise_network(
            domain,
            network_name="Delaunay CC filtered",
            edge_weight_name=None,
            edge_width=0.2,
            edge_cmap="#060606",
            add_cbar=False,
            visualise_kwargs=dict(
                objects_to_plot=("collection", "Cell centroids"),
                marker_size=0.5,
                add_cbar=True,
                color_by=("label", "Cell Type"),
                scatter_kwargs=dict(  # ← linewidths/edgecolors go HERE
                    edgecolors="none",
                ),
            ),
        )
        plt.savefig(roi_dir / f"{roi}_delaunay_cc_filtered.png")
        plt.savefig(roi_dir / f"{roi}_delaunay_cc_filtered.pdf")

        # Plot the Proximity 30μm network
        ms.visualise.visualise_network(
            domain,
            network_name="Proximity_30um",
            edge_weight_name=None,
            edge_width=0.2,
            edge_cmap="#060606",
            add_cbar=False,
            visualise_kwargs=dict(
                objects_to_plot=("collection", "Cell centroids"),
                marker_size=0.5,
                add_cbar=True,
                color_by=("label", "Cell Type"),
                scatter_kwargs=dict(  # ← linewidths/edgecolors go HERE
                    edgecolors="none",
                ),
            ),
        )
        plt.savefig(roi_dir / f"{roi}_proximity_30um.png")
        plt.savefig(roi_dir / f"{roi}_proximity_30um.pdf")

        # Calculate adjacency permutation test for the filtered Delaunay network
        print(f"Calculating adjacency permutation test for {roi} on filtered Delaunay")
        SES, SES_p_val_filtered, label_categories = (
            ms.networks.adjacency_permutation_test(
                domain,
                network_name="Delaunay CC filtered",
                label_name="Cell Type",
                alpha=0.05,
                label_shuffle_iterations=1000,
            )
        )

        ADJ_perm_dir_filtered = ADJ_perm_dir / "filtered"
        ADJ_perm_dir_filtered.mkdir(parents=True, exist_ok=True)

        ADJ_perm_dir_nonfiltered = ADJ_perm_dir / "nonfiltered"
        ADJ_perm_dir_nonfiltered.mkdir(parents=True, exist_ok=True)

        SES_df = pd.DataFrame(SES, index=label_categories, columns=label_categories)
        SES_df.to_csv(
            ADJ_perm_dir_filtered / f"filtered_adjacency_permutation_test_SES_{roi}.csv"
        )
        SES_p_val_filtered_df = pd.DataFrame(
            SES_p_val_filtered, index=label_categories, columns=label_categories
        )
        SES_p_val_filtered_df.to_csv(
            ADJ_perm_dir_nonfiltered
            / f"nonfiltered_adjacency_permutation_test_p_values_{roi}.csv"
        )

        print(f"Visualizing adjacency permutation test results for {roi}")
        ms.visualise.visualise_correlation_matrix(
            SES_p_val_filtered,
            label_categories,
            colorbar_label="Adjacency correlation (SES)",
        )
        plt.savefig(roi_dir / f"{roi}_adjacency_permutation_test.pdf")

        # Save domain
        print(f"Saving domain for {roi}...")
        ms.io.save_domain(
            domain, name_of_file=f"{roi}_muspan_domain", path_to_save=str(domains_dir)
        )
        print("Domain saved")


if __name__ == "__main__":
    main()
