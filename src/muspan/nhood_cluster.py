import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.setup_logger import setup_logger


# Base project path
paths = [
    Path("/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"),
    Path("/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/")
]

base_path = next((p for p in paths if p.exists()), None)

if base_path is None:
    raise FileNotFoundError("None of the candidate base paths exist.")

print(f"Using base path: {base_path}")


# Input
input_dir = base_path / "output" / "muspan" / "domains"

# Output directories
outpath = base_path / "output" / "muspan" / "nb_clustering"
data_dir = outpath / "data"
plots_dir = outpath / "plots"

# Create directories
for path in [outpath, data_dir, plots_dir]:
    path.mkdir(parents=True, exist_ok=True)
    
# Set up logger
logs_dir = Path(base_path) / "logs" / "muspan"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="nhood_cluster")

# Define a color palette for the neighbourhood labels
nb_colors = [
    "#5B8FA8",
    "#7EB5A6",
    "#4A7B6F",
    "#3D6B5E",
    "#89C4B0",
    "#A8C5A0",
    "#6B9E78",
    "#4E7D5B",
    "#C4956A",
    "#B07D50",
    "#D4A97A",
    "#E2C49A",
    "#8B7CB3",
    "#A08DB8",
    "#6B5E9E",
    "#B8A9CC",
    "#7A6EA8",
    "#C5B8D6",
    "#C17B6E",
    "#B56355",
    "#D4957F",
    "#A05A4A",
    "#E8B4A0",
    "#7E9E6E",
    "#92B080",
    "#5E7E50",
    "#B5C99A",
    "#6E8FAA",
    "#5A7A9A",
    "#B8A06E",
    "#9E8B5A",
    "#D4C07A",
    "#9E9E8A",
    "#7A9E9A",
    "#A0A0A0",
]

# Define variables
number_of_clusters = 10
network_type = 'Delaunay'  # 'Delaunay' or 'proximity'
max_edge_distance = 30
subset = "IPF"

# If subset is specified, create a subdirectory for plots
if subset is not None:
    plots_dir = plots_dir / subset
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
    if subset is not None and subset not in path.stem:
        logger.info(f"Skipping {path.stem} as it does not contain '{subset}' in the name.")
        continue
    logger.info(f"Loading {path.stem}...")
    domain = ms.io.load_domain(str(path))
    domain_list.append(domain)
logger.info(f"Loaded {len(domain_list)} domains from {input_dir}")


# Perform neighbourhood clustering on the dataset using KNN and minibatchkmeans
logger.info(f"Performing neighbourhood clustering with {network_type} network and {number_of_clusters} clusters...")
neighbourhood_enrichment_matrix, consistent_global_labels, unique_cluster_labels = ms.networks.cluster_neighbourhoods(
    domain_list,  # The domain dataset
    label_name='Cell Type',  # The label to use for clustering
    network_kwargs=dict(network_type=network_type, max_edge_distance=max_edge_distance, min_edge_distance=0),  # The network parameters
    k_hops=1,  # The number of hops to consider for the neighbourhood
    neighbourhood_label_name=f'Neighbourhood ID {network_type}',  # Name for the neighbourhood label
    cluster_method='minibatchkmeans',  # Clustering method
    cluster_parameters={'n_clusters': number_of_clusters,'random_state':0},  # Parameters for the clustering method
    neighbourhood_enrichment_as='log-fold' # Neighbourhood enrichment as log-fold
)

# Create a DataFrame from the neighbourhood enrichment matrix
df_ME_id = pd.DataFrame(data=neighbourhood_enrichment_matrix, index=unique_cluster_labels, columns=consistent_global_labels)
df_ME_id.index.name = f'Neighbourhood ID {network_type}'
df_ME_id.columns.name = 'Cell Type ID'

# Filter out sentinel values before computing range
logger.info("Filtering out sentinel values from the neighbourhood enrichment matrix for visualization...")
finite_vals = df_ME_id.values[np.isfinite(df_ME_id.values) & (np.abs(df_ME_id.values) < 1e300)]
vmin = np.floor(finite_vals.min())
vmax = np.ceil(finite_vals.max())
df_plot = df_ME_id.clip(lower=vmin, upper=vmax)
logger.info(f"Neighbourhood enrichment matrix value range before filtering: min={finite_vals.min()}, max={finite_vals.max()}")

# plotting vmax and vimin for the clustermap
plot_vmin = -5
plot_vmax = 5

logger.info(f"Neighbourhood enrichment matrix value range after filtering: vmin={vmin}, vmax={vmax}")
df_plot.to_csv(data_dir / f"{network_type}_{number_of_clusters}_clusters_neighbourhood_enrichment.csv")

# Visualize the neighbourhood enrichment matrix using a clustermap
logger.info("Visualizing the neighbourhood enrichment matrix using a clustermap...")
sns.clustermap(
    df_plot,
    xticklabels=consistent_global_labels,
    yticklabels=unique_cluster_labels,
    figsize=(8, 8),
    cmap='RdBu_r',
    dendrogram_ratio=(.05, .3),
    col_cluster=True,
    row_cluster=True,
    square=True,
    linewidths=0.5,
    linecolor='black',
    cbar_kws=dict(use_gridspec=False, location="top", label=f'Neighbourhood enrichment (log-fold) - khop {khop}', ticks=[plot_vmin, 0, plot_vmax]),
    cbar_pos=(0.12, 0.85, 0.72, 0.08),
    vmin=plot_vmin,
    vmax=plot_vmax,
    tree_kws={'linewidths': 1, 'color': 'black'}
)
plt.suptitle(f"{network_type.capitalize()} Neighbourhood Enrichment Clustering, {number_of_clusters} clusters)", fontsize=10)
plt.savefig(plots_dir / f"{network_type}_{number_of_clusters}_clusters_neighbourhood_heatmap.pdf", bbox_inches='tight')
plt.close()


for domain in domain_list:
    # Recolor the domain with the new neighbourhood labels
    label_name = f"Neighbourhood ID {network_type}"

    unique_labels = np.unique(domain.labels["Neighbourhood ID {network_type}"]["labels"])

    color_map = dict(zip(unique_labels, nb_colors[:len(unique_labels)]))

    domain.update_colors(
        nb_colors, colors_to_update="labels", label_name=f"Neighbourhood ID {network_type}"
    )
    
    # Set domain name
    domain_name = str(domain.name)
    
    # Visualize the domain with neighbourhood labels
    logger.info(f"Visualizing domain {domain_name} with neighbourhood labels...")
    ms.visualise.visualise(domain, color_by=f'Neighbourhood ID {network_type}', marker_size=0.5)
    plt.suptitle(f"Domain Visualization with Neighbourhood Labels for {domain_name}")
    plt.savefig(plots_dir / f"{domain_name}_{number_of_clusters}_neighbourhood_labels.pdf", bbox_inches='tight')
    plt.close()