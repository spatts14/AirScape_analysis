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
    output_dir = Path(base_dir / "density")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logs_dir = Path(base_dir) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="density")

    # Choose two cell types of interest for analysis
    cell1 = "CTHRC1+ fibroblasts"

    # # Set max
    # max_value = (
    #     100  # Set the maximum value for the color scale, will do + and - integer
    # )

    # Make a safe version of cell1 for directory naming
    cell1_safe = cell1.replace("+", "").replace(" ", "_")

    # Make directories for the cell types of interest
    cell1_dir = output_dir / cell1_safe
    cell1_dir.mkdir(parents=True, exist_ok=True)

    GDE_dir = cell1_dir / "GDE"
    GDE_dir.mkdir(parents=True, exist_ok=True)

    KDE_dir = cell1_dir / "KDE"
    KDE_dir.mkdir(parents=True, exist_ok=True)

    # Make list of all domains to process
    domain_name, domain_path = parse_args(sys.argv[1:])

    # Only name domain name
    domain_name = domain_name.replace("_muspan_domain", "")

    # Load the domain inside the worker process
    domain = ms.io.load_domain(domain_path)

    # Print the unique cell types in the domain
    logger.info(np.unique(domain.labels["Cell Type"]["labels"]))

    # Visualize the domain with cell boundaries
    logger.info(f"Visualizing KDE for {cell1}..")
    ms.distribution.kernel_density_estimation(
        domain,
        population=("Cell Type", cell1),
        contribution_label_name="Distribution values",
        visualise_output=True,
        visualise_heatmap_kwargs={
            "heatmap_cmap": "coolwarm",
            # "colorbar_limit": max_value, # does not look right
        },
    )
    plt.title(f"Cell Type: {cell1}")
    plt.tight_layout()
    plt.savefig(
        KDE_dir / f"KDE_{domain_name}.pdf",
        dpi=300,
    )

    logger.info(f"Visualizing generate distribution for {cell1}..")
    ms.distribution.generate_distribution(
        domain,
        population=("Cell Type", cell1),
        contribution_label_name="Distribution values",
        visualise_output=True,
        visualise_heatmap_kwargs={
            "heatmap_cmap": "coolwarm",
        },
    )
    plt.savefig(
        GDE_dir / f"GDE_{domain_name}.pdf",
        dpi=300,
    )

    logger.info(f"KDE and Generate Distribution Maps for {cell1}.")
    plt.close("all")
    del domain

    logger.info("Finished script!")


if __name__ == "__main__":
    main()
