"""Muspan module."""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.setup_logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args(args):
    """Parse command line arguments for the script.

    Args:
        args: List of command line arguments (excluding the script name).

    Returns:
        Tuple containing the ROI name and the path to the Xenium data directory.
    """
    parser = argparse.ArgumentParser(description="Map cell types to a domain")

    parser.add_argument(  # class for parameters
        "-r",  # shortcut
        "--roi",  # need to be the same as parameter in the domain_parallel.py file
        help="Name of the ROI being processed [required]",
        type=str,
        dest="roi",  # how you will call this variable in the code
        action="store",  # store the value provided in the command line
        required=True,
    )

    parser.add_argument(
        "-x",
        "--xenium_dir",
        help="path to the Xenium output directory for the ROI [required]",
        type=str,
        dest="xenium_dir",
        action="store",
        required=True,
    )

    results = parser.parse_args(args)
    return results.roi, results.xenium_dir


def map_cell_types_to_domain(
    cell_id_to_type_df, domain, cell_id, cluster_labels, logger
):
    """Maps cell type (cluster) labels from an df to a domain object on cell ID.

    Args:
        cell_id_to_type_df (pd.DataFrame):
            A DataFrame containing cell IDs and their corresponding types.
        domain: An object representing a spatial or logical domain, with cell IDs
            accessible via `domain.labels["Cell ID"]["labels"]` and a method
            `add_labels` for adding new labels.
        cell_id (str): The column name in `cell_id_to_type_df` that contains the cell
            IDs.
        cluster_labels (str): The column name in `cell_id_to_type_df` that contains the
            cluster or cell type labels.
        logger: A logging object for logging information and debugging.

    Returns:
        None. The function modifies the `domain` object in place by adding a
        new label with the mapped cell types.
    """
    # Get cell IDs from the domain in their original order (preserving duplicates)
    domain_cell_ids_ordered = [
        str(cell_id) for cell_id in domain.labels["Cell ID"]["labels"]
    ]

    # Get unique cell IDs for filtering cell_id_to_type_df
    domain_cell_ids_unique = set(domain_cell_ids_ordered)

    logger.info(f"Number of unique cells in the domain: {len(domain_cell_ids_unique)}")
    logger.info(
        f"Total cell entries in domain (including duplicates): "
        f"{len(domain_cell_ids_ordered)}"
    )

    # Filter cell_id_to_type_df to include only cells in the area of interest
    filt_cell_id_to_type_df = cell_id_to_type_df[
        cell_id_to_type_df[cell_id].isin(domain_cell_ids_unique)
    ]

    logger.info(
        f"Filtered cell_id_to_type_df from {len(cell_id_to_type_df)}"
        f" to {len(filt_cell_id_to_type_df)} cells"
    )

    # Add cell cluster IDs
    logger.info("Adding cell_type IDs to domain with cluster labels")

    # Create a mapping from cell_id to cell_type on filtered data
    cell_id_to_type = dict(
        zip(
            filt_cell_id_to_type_df[cell_id],
            filt_cell_id_to_type_df[cluster_labels],
        )
    )

    # Get cell types in the same order as domain cell IDs
    cell_types_ordered = [
        cell_id_to_type.get(cell_id, "Unknown") for cell_id in domain_cell_ids_ordered
    ]

    # Add cell_type label to the domain
    domain.add_labels(label_name="Cell Type", labels=cell_types_ordered)

    logger.info(f"Label keys in domain: {domain.labels.keys()}")
    logger.info(f"Length of cell_types_ordered: {len(cell_types_ordered)}")
    logger.info(f"Length of domain cell IDs: {len(domain_cell_ids_ordered)}")
    logger.info(
        f"Number of 'Unknown' cell types: {cell_types_ordered.count('Unknown')}"
    )

    # Return muspan domain with cell types mapped
    return domain


def main():
    """Main function to create and save a MuSpAn domain for a given ROI."""
    (
        roi,
        xenium_dir,
    ) = parse_args(
        sys.argv[1:]
    )  # parse command line arguments, excluding the script name

    # Set paths
    base_dir = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )

    out_dir = base_dir / "output" / "muspan"
    domains_dir = out_dir / "domains"
    figs_dir = out_dir / "figures"
    ADJ_perm_dir = out_dir / "adjacency_permutation_test_results"

    # Make directories if they don't exist
    dir_list = [out_dir, domains_dir, figs_dir, ADJ_perm_dir]
    for directory in dir_list:
        directory.mkdir(parents=True, exist_ok=True)

    # Set up logger
    wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/"
    logs_dir = Path(wd) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="muspan")

    # Set colors for cell types
    level_2_palette = {
        # Epithelial
        "Ciliated cells": "#5B8FA8",
        "Goblet cells": "#7EB5A6",
        "Basal cells": "#4A7B6F",
        "Proliferating Basal cells": "#3D6B5E",
        "Secretory epithelial cells": "#89C4B0",
        "AT1 cells": "#A8C5A0",
        "AT2 cells": "#6B9E78",
        "Proliferating AT2 cells": "#4E7D5B",
        # Fibroblasts
        "Adventitial fibroblasts": "#C4956A",
        "CTHRC1+ fibroblasts": "#B07D50",
        "Alveolar fibroblasts": "#D4A97A",
        "Lipo-fibroblasts": "#E2C49A",
        # Endothelial
        "Lymphatic endothelial cells": "#8B7CB3",
        "Pulmonary vein endothelial cell": "#A08DB8",
        "Blood endothelial cells - unclassified": "#6B5E9E",
        "Capillary endothelial cells": "#B8A9CC",
        "Pulmonary artery endothelial cell": "#7A6EA8",
        "Pericytes": "#C5B8D6",
        # Macrophages & monocytes
        "Macrophages": "#C17B6E",
        "Lipid-associated macrophages": "#B56355",
        "Interstitial Macrophages": "#D4957F",
        "Airway/Alveolar macrophages": "#A05A4A",
        "Monocytes": "#E8B4A0",
        # T cells & NK cells
        "T cells": "#7E9E6E",
        "CD4+ T cells": "#92B080",
        "CD8+ T cells": "#5E7E50",
        "NK cells": "#B5C99A",
        # Other immune
        "B cells": "#6E8FAA",
        "Plasma cells": "#5A7A9A",
        "Dendritic cells": "#B8A06E",
        "Mast cells": "#9E8B5A",
        "Neutrophils": "#D4C07A",
        # Other
        "SMC": "#9E9E8A",
        "Aerocytes": "#7A9E9A",
        "Unknown": "#A0A0A0",
    }

    # Set variables
    transcripts_to_load = [
        "EPCAM",
        "VWF",
        "ACTA2",
    ]
    cell_id = "cell_id"
    cluster_labels = "level_2"

    # Load cell id and cell type dictionary
    cell_id_to_type_df = pd.read_csv(
        base_dir / "output/muspan/cell_id_to_cluster_labels.csv"
    )

    # Make list of all paths to Xenium files
    logger.info("Loading MuSpAn data...")

    # Create muspan object
    logger.info(f"Processing ROI: {roi} with Xenium data at {xenium_dir}")

    # Calculate time to create and save domain for this ROI
    start_time = pd.Timestamp.now()

    # Make fig directory for this ROI domain
    roi_dir = figs_dir / roi
    roi_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating MuSpAn domain object...")
    domain = ms.io.xenium_to_domain(
        path_to_xenium_data=str(xenium_dir),  # path to ROI
        domain_name=roi,  # ROI name
        load_transcripts=True,  # load transcripts, but only the selected ones
        selected_transcripts=transcripts_to_load,  # list of transcripts to load
        load_nuclei=True,  # load nuclei boundaries
        load_cells_as_shapes=True,  # load cell boundaries as shapes
        exclude_no_nuclei_cells=True,  # exclude cells without nuclei from the domain
        add_transcript_counts_to_cell=False,  # do not add transcript counts to labels
    )
    logger.info(f"Domain object created: {domain}")

    # Add cell type IDs to domain in label called "Cell Type"
    logger.info("Mapping cell types to domain based on cell_id to Cell Type")
    domain = map_cell_types_to_domain(
        cell_id_to_type_df, domain, cell_id, cluster_labels, logger
    )
    # Confirm cell type mapping was successful
    if "Cell Type" in domain.labels:
        logger.info("Cell Type label successfully added to domain")
    else:
        logger.warning("Cell Type label not found in domain after mapping")

    # Show the unique cell types that were mapped
    unique_cell_types = set(domain.labels["Cell Type"]["labels"])
    logger.info(f"Unique cell types mapped to domain: {unique_cell_types}")

    # Convert cell boundaries to cell centers (centroids)
    logger.info("Convert cell boundaries to cell centers (centroids)")
    domain.convert_objects(
        population=("Collection", "Cell boundaries"),
        object_type="point",
        conversion_method="centroids",
        collection_name="Cell centroids",
        inherit_collections=False,
    )

    # Query to isolate Cell centroids for visualization
    logger.info("Querying domain to isolate cell centroids for visualization")
    boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")
    centCells = ms.query.query(domain, ("Collection",), "is", "Cell centroids")

    # Update colors
    domain.update_colors(
        level_2_palette, colors_to_update="labels", label_name="Cell Type"
    )

    # Visualize Cell Types
    logger.info(
        f"Visualize the MuSpAn domain for {roi} with cell types and cell boundaries"
    )
    ms.visualise.visualise(
        domain,
        color_by=("label", "Cell Type"),
        objects_to_plot=boundCells,
        shape_kwargs=dict(alpha=1, linewidth=0.01, edgecolor="#00000000"),
    )
    plt.savefig(roi_dir / f"{roi}_cell_types_boundaries.png")
    plt.savefig(roi_dir / f"{roi}_cell_types_boundaries.pdf")

    logger.info(
        f"Visualize the MuSpAn domain for {roi} with cell types and cell centroids"
    )
    ms.visualise.visualise(
        domain,
        color_by=("label", "Cell Type"),
        objects_to_plot=centCells,
        marker_size=0.5,
    )
    plt.savefig(roi_dir / f"{roi}_cell_types_centroids.png")
    plt.savefig(roi_dir / f"{roi}_cell_types_centroids.pdf")

    logger.info("Generating spatial networks for the domain...")
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

    logger.info(f"Visualize the networks for {roi}")
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
    logger.info(
        f"Calculating adjacency permutation test for {roi} on filtered Delaunay network"
    )
    SES, SES_p_val_filtered, label_categories = ms.networks.adjacency_permutation_test(
        domain,
        network_name="Delaunay CC filtered",
        label_name="Cell Type",
        alpha=0.05,
        label_shuffle_iterations=1000,
    )

    SES_p_val_filtered_df = pd.DataFrame(
        SES_p_val_filtered, index=label_categories, columns=label_categories
    )
    SES_p_val_filtered_df.to_csv(
        ADJ_perm_dir / f"adjacency_permutation_test_p_values_{roi}.csv"
    )

    logger.info(f"Visualizing adjacency permutation test results for {roi}")
    ms.visualise.visualise_correlation_matrix(
        SES_p_val_filtered,
        label_categories,
        colorbar_label="Adjacency correlation (SES)",
    )
    plt.savefig(roi_dir / f"{roi}_adjacency_permutation_test.pdf")

    # Save domain
    logger.info(f"Saving domain for {roi}...")
    ms.io.save_domain(
        domain, name_of_file=f"{roi}_muspan_domain", path_to_save=str(domains_dir)
    )
    logger.info("Domain saved")

    # Calculate and log time taken to create and save domain for this ROI
    end_time = pd.Timestamp.now()
    time_taken = end_time - start_time
    logger.info(f"Time taken to create and save domain for {roi}: {time_taken}")

    logger.info("All domains saved")


if __name__ == "__main__":
    main()
