"""Calculate neighbourhood clusters (fixed at 18) and save the annotated domains to disk."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

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


def main():
    """Compute neighbourhood clustering at a fixed cluster count and save the domains."""
    # Define variables
    number_of_clusters = 18
    khop = 1  # Number of hops for neighbourhood clustering
    network_type = "proximity"  # 'Delaunay' or 'proximity'
    max_edge_distance = 30
    subset = ["IPF", "PM08"]  # COPD or IPF and PM08
    subset_safe_name = "v".join(subset)
    subset_safe_name = f"{subset_safe_name}_159removed"

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

    # Output directory for the annotated domains
    outpath = base_path / "output" / "muspan" / "nb_clustering"
    domains_out_dir = (
        outpath
        / "domains_with_niches"
        / network_type
        / subset_safe_name
        / f"{number_of_clusters}_clusters"
    )
    domains_out_dir.mkdir(parents=True, exist_ok=True)

    # Make cluster number of clusters
    plots_dir_cluster = domains_out_dir / "plots"
    plots_dir_cluster.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logs_dir = Path(base_path) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        log_dir=logs_dir, log_name=f"nhood_cluster_save_{number_of_clusters}"
    )

    # Set color scheme
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

    # Load the domains
    domain_list = []

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

    # Remove IPF_RBH_159 domain from the list
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

    # Perform neighbourhood clustering on the dataset using KNN and minibatchkmeans
    logger.info(
        f"Performing neighbourhood clustering with {network_type} network and"
        f" {number_of_clusters} clusters..."
    )
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
    logger.info("Clustering complete. Niche labels have been added to each domain.")

    # Update the colors of the neighbourhood labels in each domain based on the unique labels
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

        # Get cell centroids for plotting
        qCells = ms.query.query(domain, ("Collection",), "is", "Cell centroids")

        # Set domain name
        domain_name = str(domain.name)

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

    # Save each domain, now annotated with the neighbourhood labels, to the new folder
    logger.info(f"Saving annotated domains to {domains_out_dir}...")
    for domain in domain_list:
        domain_name = str(domain.name)
        save_path = domains_out_dir / f"{domain_name}.muspan"
        ms.io.save_domain(domain, str(save_path))
        logger.info(f"Saved {domain_name} to {save_path}")

    logger.info(f"Finished saving {len(domain_list)} annotated domains.")


if __name__ == "__main__":
    main()
