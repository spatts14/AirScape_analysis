"""Visualize cell types and compute TCM for a given domain."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.setup_logger import setup_logger


def parse_args(args):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute cross-PCF for a domain")

    parser.add_argument(
        "-dn",
        "--domain_name",
        help="Name of the domain being processed [required]",
        type=str,
        dest="domain_name",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--domain",
        help="Path to the .muspan domain file [required]",
        type=str,
        dest="domain_path",  # renamed to make clear it's a path
        required=True,
    )

    results = parser.parse_args(args)
    return results.domain_name, results.domain_path


def main():
    """Main function to visualize cell types and compute TCM for a given domain."""
    # Set directory paths
    base_dir = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/muspan"
    )
    output_dir = Path(base_dir / "manual_visualizations")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logs_dir = Path(base_dir) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="vis_TCM")

    # Choose two cell types of interest for analysis
    cell1 = "Plasma cells"
    cell2 = "Plasma cells"

    # Make cell type names safe for filenames
    cell1_safe = cell1.replace("/", "_").replace(" ", "_")
    cell2_safe = cell2.replace("/", "_").replace(" ", "_")
    clusters_of_interest = [cell1, cell2]

    # Make list of all domains to process
    domain_name, domain_path = parse_args(sys.argv[1:])

    # Only name domain name
    domain_name = domain_name.replace("_muspan_domain", "")

    # Load the domain inside the worker process
    domain = ms.io.load_domain(domain_path)

    # make domain directory for saving visualizations
    domain_output_dir = output_dir / domain_name
    domain_output_dir.mkdir(parents=True, exist_ok=True)

    # Print the unique cell types in the domain
    logger.info(np.unique(domain.labels["Cell Type"]["labels"]))

    # Boundaries of cells
    boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    # Level 1
    logger.info("Visualizing level 1 cell types...")
    ms.visualise.visualise(
        domain,
        color_by=("label", "Cell Type level 1"),
        objects_to_plot=boundCells,
        shape_kwargs=dict(alpha=1, linewidth=0.01, edgecolor="#00000000"),
        add_scalebar=True,
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
    plt.savefig(
        domain_output_dir / "level_1_cell_types_boundaries.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.savefig(
        domain_output_dir / "level_1_cell_types_boundaries.pdf",
        bbox_inches="tight",
        dpi=600,
    )

    # Level 2
    logger.info("Visualizing level 2 cell types...")
    ms.visualise.visualise(
        domain,
        color_by=("label", "Cell Type"),
        objects_to_plot=boundCells,
        shape_kwargs=dict(alpha=1, linewidth=0.01, edgecolor="#00000000"),
        add_scalebar=True,
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
    plt.savefig(
        domain_output_dir / "level_2_cell_types_boundaries.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.savefig(
        domain_output_dir / "level_2_cell_types_boundaries.pdf",
        bbox_inches="tight",
        dpi=600,
    )

    # 'Interstitial macrophages', 'Ciliated cells'
    cluster_of_interest_query = ms.query.query(
        domain, ("label", "Cell Type"), "in", clusters_of_interest
    )

    # Update color if needed
    # Set color for interstitial macrophages
    # domain.update_colors(
    #     {"Interstitial macrophages": "#f7b7d2"},
    #     colors_to_update="labels",
    #     label_name="Cell Type",
    # )

    # Visualize the domain with cell boundaries
    bound_cells_query = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    # Visualize the domain with cell boundaries
    logger.info("Visualizing the domain with cell boundaries...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ms.visualise.visualise(
        domain,
        objects_to_plot=bound_cells_query,
        add_cbar=False,
        shape_kwargs={
            "alpha": 0.5,
            "linewidth": 0.005,
            "edgecolor": "#00000000",
            "color": "#e5e3c4",
        },
        ax=ax,
    )

    ms.visualise.visualise(
        domain,
        objects_to_plot=cluster_of_interest_query,
        color_by="Cell Type",
        ax=ax,
        marker_size=4,
        shape_kwargs={"linewidth": 0.4, "alpha": 1},
        add_scalebar=True,
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
    plt.title(f"Cell Types: {cell1} and {cell2}")
    plt.tight_layout()
    plt.savefig(
        domain_output_dir / f"cell_types_{cell1_safe}_{cell2_safe}_{domain.name}.png",
        dpi=300,
    )

    # Calculate TCM
    # compute and visualise the topographical correlation map between points
    logger.info(f"Calculating TCM between {cell1} and {cell2}...")
    TCM_array = ms.spatial_statistics.topographical_correlation_map(
        domain,
        population_A=("Cell Type", cell1),
        population_B=("Cell Type", cell2),
        mesh_step=5,
        radius_of_interest=50,
        kernel_radius=150,
        kernel_sigma=30,
        visualise_output=False,
    )

    # Visualize TCM
    logger.info(f"Visualizing TCM between {cell1} and {cell2}...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ms.visualise.visualise(
        domain,
        objects_to_plot=bound_cells_query,
        add_cbar=False,
        shape_kwargs={
            "alpha": 0.5,
            "linewidth": 0.005,
            "edgecolor": "#00000000",
            "color": "#bec1c2",
        },
        add_scalebar=True,
        scalebar_kwargs={
            "size": 1000,
            "label": "1000µm",
            "loc": "lower right",
            "pad": 0.1,
            "color": "black",
            "frameon": False,
            "size_vertical": 2,
        },
        ax=ax,
    )

    ms.visualise.visualise_topographical_correlation_map(
        domain,
        TCM_array,
        ax=ax,
        colorbar_limit=None,
        tcm_cmap="RdBu_r",
        colorbar_label="TCM",
    )
    plt.title(f"Cell Types: {cell1} and {cell2}")
    plt.tight_layout()
    plt.savefig(
        domain_output_dir
        / f"cell_types_{cell1_safe}_{cell2_safe}_{domain.name}_TCM.png",
        dpi=300,
    )

    logger.info(f"TCM calculation and visualization completed for {cell1} and {cell2}.")
    plt.close("all")
    del domain

    logger.info("Finished script!")


if __name__ == "__main__":
    main()
