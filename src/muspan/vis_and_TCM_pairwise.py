"""Visualize cell types and compute TCM for all pairwise cell-type interactions."""

import argparse
import itertools
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


def level_1_vis(domain, domain_output_dir, boundCells, logger):
    """Visualize level 1 cell types in the domain.

    Args:
        domain: The muspan domain object.
        domain_output_dir: Path to the output directory for saving visualizations.
        boundCells: Query result for cell boundaries in the domain.
        logger: Logger object for logging messages.

    Returns:
        None
    """
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


def level_2_vis(domain, domain_output_dir, boundCells, logger):
    """Visualize level 2 cell types in the domain.

    Args:
        domain: The muspan domain object.
        domain_output_dir: Path to the output directory for saving visualizations.
        boundCells: Query result for cell boundaries in the domain.
        logger: Logger object for logging messages.

    Returns:
        None
    """
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


def process_pair(
    domain, cell1, cell2, output_dir, domain_name, boundCells, vmax, logger
):
    """Compute and visualize the TCM between one pair of cell types (may be cell1 == cell2).

    Args:
        domain: The muspan domain object (already loaded, colors already updated).
        cell1: First cell type of interest.
        cell2: Second cell type of interest.
        output_dir: Base output directory (TCM_visualizations).
        domain_name: Name of the domain (used in output filenames).
        boundCells: Query result for cell boundaries in the domain (domain-wide, precomputed).
        vmax: Color scale limit for the TCM plot.
        logger: Logger object for logging messages.

    Returns:
        None
    """
    clusters_of_interest = [cell1, cell2]
    clusters_of_interest_name = (
        "_".join(clusters_of_interest).replace("/", "_").replace(" ", "_")
    )

    clusters_of_interest_dir = output_dir / clusters_of_interest_name
    clusters_of_interest_dir.mkdir(parents=True, exist_ok=True)

    # Cell types selected for downstream plotting and TCM calculation
    cluster_of_interest_query = ms.query.query(
        domain, ("label", "Cell Type"), "in", clusters_of_interest
    )

    # Visualize the domain with cell boundaries
    logger.info(
        f"Visualizing the domain with cell boundaries for {cell1} vs {cell2}..."
    )
    _, ax = plt.subplots(figsize=(10, 5))
    ms.visualise.visualise(
        domain,
        objects_to_plot=boundCells,
        add_cbar=False,
        shape_kwargs={
            "alpha": 0.5,
            "linewidth": 0.005,
            "edgecolor": "#00000000",
            "color": "#707374",
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
    plt.tight_layout()
    plt.savefig(
        clusters_of_interest_dir / f"{domain_name}.png",
        dpi=300,
    )
    plt.close()

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
    _, ax = plt.subplots(figsize=(10, 8))
    ms.visualise.visualise(
        domain,
        objects_to_plot=boundCells,
        add_cbar=False,
        shape_kwargs={
            "alpha": 0.5,
            "linewidth": 0.005,
            "edgecolor": "#00000000",
            "color": "#707374",
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
        colorbar_limit=vmax,
        tcm_cmap="RdBu_r",
        colorbar_label="TCM",
    )
    plt.tight_layout()
    plt.savefig(
        clusters_of_interest_dir / f"{domain_name}_TCM.png",
        dpi=600,
    )
    plt.close()

    logger.info(f"TCM calculation and visualization completed for {cell1} and {cell2}.")


def main():
    """Main function to visualize cell types and compute TCM for all pairwise interactions."""
    # Set directory paths
    base_dir = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/muspan"
    )
    output_dir = Path(base_dir / "TCM_visualizations")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logs_dir = Path(base_dir) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="vis_TCM")

    vmax = 30  # Set the maximum value for the color scale, will do + and - integer

    # Visualization levels
    vis_levels = False  # Set to True to visualize level 1 and level 2

    # Set to a cell type name (e.g. "CD4+ T cells") to only compute that cell type against
    # every other cell type (including itself). Set to None to compute the full pairwise
    # sweep across all cell types instead.
    FIXED_CELL_TYPE_OF_INTEREST = "CD4+ T cells"

    # Make list of all domains to process
    domain_name, domain_path = parse_args(sys.argv[1:])

    # Only name domain name
    domain_name = domain_name.replace("_muspan_domain", "")

    # Load the domain inside the worker process
    domain = ms.io.load_domain(domain_path)

    # Print the unique cell types in the domain
    cell_types = sorted(np.unique(domain.labels["Cell Type"]["labels"]))
    logger.info(f"Cell types found in domain: {cell_types}")

    # Boundaries of cells (domain-wide, reused for every pair)
    boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    # Level 1 / level 2 visualizations are domain-wide (not pair-specific) — run once,
    # not once per pair.
    if vis_levels:
        domain_output_dir = output_dir / domain_name
        domain_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Visualizing level 1...")
        level_1_vis(domain, domain_output_dir, boundCells, logger)

        logger.info("Visualizing level 2...")
        level_2_vis(domain, domain_output_dir, boundCells, logger)

    # Update color if needed (domain-wide, done once)
    # domain.update_colors(
    #     {"T cells": # cell typr
    # "#C8A7E3" # color
    # },
    #     colors_to_update="labels",
    #     label_name="Cell Type",
    # )

    if FIXED_CELL_TYPE_OF_INTEREST is not None:
        # Only pairs involving the fixed cell type, including itself — e.g. for
        # FIXED_CELL_TYPE_OF_INTEREST = "CD4+ T cells" and cell_types = ["A", "B", "CD4+ T cells"]
        # this gives ("CD4+ T cells","A"), ("CD4+ T cells","B"), ("CD4+ T cells","CD4+ T cells").
        if FIXED_CELL_TYPE_OF_INTEREST not in cell_types:
            raise ValueError(
                f"FIXED_CELL_TYPE_OF_INTEREST '{FIXED_CELL_TYPE_OF_INTEREST}' not found in "
                f"domain cell types: {cell_types}"
            )
        cell_type_pairs = [(FIXED_CELL_TYPE_OF_INTEREST, other) for other in cell_types]
    else:
        # Every unordered pair of cell types, including self-pairs (A vs A) — e.g. for
        # cell_types = ["A", "B", "C"] this gives ("A","A"), ("A","B"), ("A","C"),
        # ("B","B"), ("B","C"), ("C","C"). Does NOT compute ("B","A") separately from ("A","B").
        cell_type_pairs = list(itertools.combinations_with_replacement(cell_types, 2))

    logger.info(f"Processing {len(cell_type_pairs)} cell-type pairs...")

    for i, (cell1, cell2) in enumerate(cell_type_pairs, start=1):
        logger.info(f"[{i}/{len(cell_type_pairs)}] {cell1} vs {cell2}")
        try:
            process_pair(
                domain=domain,
                cell1=cell1,
                cell2=cell2,
                output_dir=output_dir,
                domain_name=domain_name,
                boundCells=boundCells,
                vmax=vmax,
                logger=logger,
            )
        except Exception:
            # Log and continue so one problematic pair (e.g. too few cells of one type)
            # doesn't kill the whole batch — re-raise here if you'd rather fail fast instead.
            logger.exception(f"Failed processing {cell1} vs {cell2} — skipping.")
        finally:
            plt.close("all")

    del domain

    logger.info("Finished script!")


if __name__ == "__main__":
    main()
