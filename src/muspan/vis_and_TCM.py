from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import muspan as ms

# Set directory paths
base_dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/muspan"
)
data_dir = Path(base_dir / "domains")
output_dir = Path(base_dir / "manual_visualizations")

# Make sure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# List all domain of domain
## SHAM
# COPD_R009
# COPD_R021
# COPD_R025
# COPD_R036
# COPD_R039
# COPD_R041

## TREATMENT
list = [
    "COPD_R039_V1"
    # "IPF_RBH_16",
    # "PM08_167",
    # "MICA_III_319_315_311",
    # "COPD_R011_V1",
    # "COPD_46005_V1",
    # "COPD_R003_V1",
    # "COPD_R014_V1",
    # "COPD_R035_V1",
    # "COPD_R038_V1",
    # "COPD_R011_V2",
    # "COPD_46005_V2",
    # "COPD_R003_V2",
    # "COPD_R014_V2",
    # "COPD_R035_V2",
    # "COPD_R038_V2",
    # "COPD_R009_V2",
    # "COPD_R021_V2",
    # "COPD_R025_V2",
    # "COPD_R036_V2",
    # "COPD_R039_V2",
    # "COPD_R041_V2",
]

for domain_name in list:
    print(f"Processing domain: {domain_name}")

    # Load domain
    domain = ms.io.load_domain(str(data_dir / f"{domain_name}_muspan_domain.muspan"))

    # make domain directory for saving visualizations
    domain_output_dir = output_dir / domain.name
    domain_output_dir.mkdir(parents=True, exist_ok=True)

    # Print the unique cell types in the domain
    print(np.unique(domain.labels["Cell Type"]["labels"]))

    # Boundaries of cells
    boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    # Level 1
    print("Visualizing level 1 cell types...")
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
    print("Visualizing level 2 cell types...")
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

    # Choose two cell types of interest for analysis
    cell1 = "Interstitial macrophages"
    cell2 = "Interstitial macrophages"
    clusters_of_interest = [cell1, cell2]
    # 'Interstitial macrophages', 'Ciliated cells'
    cluster_of_interest_query = ms.query.query(
        domain, ("label", "Cell Type"), "in", clusters_of_interest
    )

    # Update color if needed
    # Set color for interstitial macrophages
    domain.update_colors(
        {"Interstitial macrophages": "#f7b7d2"},
        colors_to_update="labels",
        label_name="Cell Type",
    )

    bound_cells_query = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    # Visualize the domain with cell boundaries
    print("Visualizing the domain with cell boundaries...")
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
        domain_output_dir / f"cell_types_{cell1}_{cell2}_{domain.name}.png", dpi=300
    )

    # Calculate TCM
    # compute and visualise the topographical correlation map between points
    print(f"Calculating TCM between {cell1} and {cell2}...")
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
    print(f"Visualizing TCM between {cell1} and {cell2}...")
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
        domain_output_dir / f"cell_types_{cell1}_{cell2}_{domain.name}_TCM.png", dpi=300
    )

    print(f"TCM calculation and visualization completed for {cell1} and {cell2}.")
    plt.close("all")
    del domain

print("Finished!")
