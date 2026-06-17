"""Muspan module."""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.setup_logger import setup_logger
from utils.airspace_colors import level_1_palette, level_2_palette

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
    cell_id_to_type_df,
    domain,
    cell_id,
    label_name="Cell Type",
    cluster_labels=None,
    logger=None,
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
        label_name (str): The name of the label to be added to the domain for cell types.
            Default is "Cell Type".
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
    domain.add_labels(label_name=label_name, labels=cell_types_ordered)

    logger.info(f"Label keys in domain: {domain.labels.keys()}")
    logger.info(f"Length of cell_types_ordered: {len(cell_types_ordered)}")
    logger.info(f"Length of domain cell IDs: {len(domain_cell_ids_ordered)}")
    logger.info(
        f"Number of 'Unknown' cell types: {cell_types_ordered.count('Unknown')}"
    )

    # Return muspan domain with cell types mapped
    return domain


def filter_cell_types(domain, roi, logger):
    """Filter unwanted cell types from a domain based on the ROI condition.

    Uses domain.delete_objects() to remove spatial objects AND their associated
    labels simultaneously, keeping the domain internally consistent.
    """

    CELLS_TO_REMOVE_ALL = ["Unknown", "nan"]

    COPD_CELLS_TO_REMOVE = [
        "AT1 cells",
        "AT2 cells",
        "Proliferating AT2 cells",
        "Airway/Alveolar macrophages",
        "Alveolar fibroblasts (collagen high)",
        "Alveolar fibroblasts",
        "Lipid-associated macrophages",
    ]

    cells_to_remove = list(CELLS_TO_REMOVE_ALL)
    if "COPD" in roi:
        cells_to_remove.extend(COPD_CELLS_TO_REMOVE)
        logger.info(
            f"[{roi}] COPD domain detected — applying COPD-specific filtering "
            f"in addition to universal filtering."
        )
    else:
        logger.info(
            f"[{roi}] Non-COPD domain — applying universal filtering only "
            f"(Unknown + nan)."
        )

    # Log before state
    cell_type_labels = domain.labels["Cell Type"]["labels"]
    types_before = sorted(set(cell_type_labels))
    n_before = domain.n_objects
    types_being_removed = [ct for ct in cells_to_remove if ct in types_before]

    logger.info(f"[{roi}] Objects BEFORE filtering: {n_before}")
    logger.info(f"[{roi}] Cell types being removed: {types_being_removed}")

    # Build query for objects TO REMOVE using the muspan query API
    query_remove = ms.query.query(
        domain, ("label", "Cell Type"), "in list", cells_to_remove
    )

    # Delete objects using muspan's own method
    # This removes spatial objects AND performs label bookkeeping automatically
    domain.delete_objects(query_remove)

    # Log after state
    n_after = domain.n_objects
    types_after = sorted(set(domain.labels["Cell Type"]["labels"]))
    n_removed = n_before - n_after

    logger.info(f"[{roi}] Objects AFTER filtering: {n_after} ({n_removed} removed)")
    logger.info(f"[{roi}] Remaining cell types: {types_after}")

    # Verify none of the removed types remain
    remaining_removed = [ct for ct in cells_to_remove if ct in types_after]
    if remaining_removed:
        logger.warning(
            f"[{roi}] WARNING — these types were not fully removed: {remaining_removed}"
        )
    else:
        logger.info(
            f"[{roi}] Filtering verified — all target cell types successfully removed."
        )

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
    wd = "/rds/general/user/sep22/home/Projects/AirScape_analysis/HPC_jobs/"
    logs_dir = Path(wd) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="muspan")

    # Set variables
    transcripts_to_load = ["KRT5", "VWF", "ACTA2", "MRC1", "CD4", "CD8A", "16S"]
    cell_id = "cell_id"

    # Load cell id and cell type dictionary
    cell_id_to_type_df_level1 = pd.read_csv(
        base_dir / "output/muspan/cell_id_to_cluster_labels_level1.csv"
    )
    cell_id_to_type_df_level2 = pd.read_csv(
        base_dir / "output/muspan/cell_id_to_cluster_labels_level2.csv"
    )
    cell_id_to_type_df_level3 = pd.read_csv(
        base_dir / "output/muspan/cell_id_to_cluster_labels_level3.csv"
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

    # Map level 1
    logger.info("Mapping level 1 to Cell Type level 1 label in domain")
    domain = map_cell_types_to_domain(
        cell_id_to_type_df_level1,
        domain,
        cell_id=cell_id,
        cluster_labels="level_1",
        label_name="Cell Type level 1",
        logger=logger,
    )

    # Confirm cell type mapping was successful
    if "Cell Type level 1" in domain.labels:
        logger.info("Cell Type level 1 label successfully added to domain")
    else:
        logger.warning("Cell Type level 1 label not found in domain after mapping")

    # Show the unique cell types that were mapped
    unique_cell_types = set(domain.labels["Cell Type level 1"]["labels"])
    logger.info(f"Unique cell types mapped to domain: {unique_cell_types}")

    # Map level 2
    logger.info("Mapping level 2 to Cell Type label in domain")
    domain = map_cell_types_to_domain(
        cell_id_to_type_df_level2,
        domain,
        cell_id=cell_id,
        cluster_labels="level_2",
        label_name="Cell Type",
        logger=logger,
    )

    # Confirm cell type mapping was successful
    if "Cell Type" in domain.labels:
        logger.info("Cell Type label successfully added to domain")
    else:
        logger.warning("Cell Type label not found in domain after mapping")

    # Show the unique cell types that were mapped
    unique_cell_types = set(domain.labels["Cell Type"]["labels"])
    logger.info(f"Unique cell types mapped to domain: {unique_cell_types}")

    # Map level 3
    logger.info("Mapping level 3 to Cell Type level 3 label in domain")
    domain = map_cell_types_to_domain(
        cell_id_to_type_df_level3,
        domain,
        cell_id=cell_id,
        cluster_labels="level_3",
        label_name="Cell Type level 3",
        logger=logger,
    )

    # Update colors
    domain.update_colors(
        level_2_palette, colors_to_update="labels", label_name="Cell Type"
    )

    domain.update_colors(
        level_1_palette, colors_to_update="labels", label_name="Cell Type level 1"
    )

    # Filter unwanted cell types from the domain based on the ROI condition
    logger.info(f"Filtering unwanted cell types from domain for {roi}...")
    domain = filter_cell_types(domain, roi, logger)

    # Save the filtered domain
    logger.info(f"Saving filtered domain for {roi}...")
    ms.io.save_domain(
        domain, name_of_file=f"{roi}_muspan_domain", path_to_save=str(domains_dir)
    )
    logger.info("Filtered domain saved")

    # Reload the saved filtered domain
    # Reload so all subsequent muspan operations use the saved filtered file
    logger.info(f"Reloading saved filtered domain for {roi}...")
    saved_domain_path = domains_dir / f"{roi}_muspan_domain.muspan"
    domain = ms.io.load_domain(str(saved_domain_path))
    logger.info(f"Reloaded domain: {domain}")

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

    # Visualize Cell Types
    logger.info(
        f"Visualize the MuSpAn domain for {roi} with cell types and cell boundaries"
    )
    ms.visualise.visualise(
        domain,
        color_by=("label", "Cell Type"),
        objects_to_plot=boundCells,
        shape_kwargs=dict(alpha=1, linewidth=0.01, edgecolor="#00000000"),
        scalebar_kwargs={
            "size": 1000,
            "label": "1000µm",
            "loc": "lower right",
            "pad": 0.1,
            "color": "black",
            "frameon": False,
            "size_vertical": 2,
        },
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
        scalebar_kwargs={
            "size": 1000,
            "label": "1000µm",
            "loc": "lower right",
            "pad": 0.1,
            "color": "black",
            "frameon": False,
            "size_vertical": 2,
        },
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
        scalebar_kwargs={
            "size": 1000,
            "label": "1000µm",
            "loc": "lower right",
            "pad": 0.1,
            "color": "black",
            "frameon": False,
            "size_vertical": 2,
        },
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
        scalebar_kwargs={
            "size": 1000,
            "label": "1000µm",
            "loc": "lower right",
            "pad": 0.1,
            "color": "black",
            "frameon": False,
            "size_vertical": 2,
        },
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
