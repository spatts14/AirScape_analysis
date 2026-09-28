"""Visualize all clusters in a domain — parallelizable via PBS job array."""

from pathlib import Path

import matplotlib.pyplot as plt

import muspan as ms


def main():
    """Visualize clusters for a single domain."""
    network_type = "proximity"  # "Delaunay" or "proximity"
    niche_label_name = f"Neighbourhood ID {network_type}"
    dir_name = "PM08vIPF_khop_1"
    clusters_of_interest = [12, 13, 3]
    cluster = 18
    sample_ID = "IPF_RBH_16_proximity_18_muspan_domain.muspan"

    base_path = Path(
        "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
    )

    domain_dir = (
        base_path
        / "output"
        / "muspan"
        / "nb_clustering"
        / "domains_with_niches"
        / network_type
        / dir_name
        / f"{cluster}_clusters"
    )
    domain_path = domain_dir / sample_ID

    if not domain_path.exists():
        raise FileNotFoundError(f"Domain file not found: {domain_path}")

    save_path = domain_dir / "plots" / "manual"
    save_path.mkdir(parents=True, exist_ok=True)

    domain = ms.io.load_domain(str(domain_path))
    boundCells = ms.query.query(domain, ("Collection",), "is", "Cell boundaries")

    save_path_domain = save_path / str(domain.name.replace(".muspan", ""))
    save_path_domain.mkdir(parents=True, exist_ok=True)

    # Update color
    domain.update_colors(
        {17: "#8B7CB3"}, colors_to_update="labels", label_name=niche_label_name
    )

    # --- Plot combined clusters of interest, before the per-cluster loop
    # below reassigns selected_boundaries ---
    label = "_".join(str(c) for c in clusters_of_interest)

    selected_clusters = ms.query.query(
        domain, ("label", niche_label_name), "in", clusters_of_interest
    )
    selected_boundaries = selected_clusters & boundCells

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
    plt.savefig(
        f"{save_path_domain}/MANUAL_clusters_{label}.pdf", bbox_inches="tight", dpi=300
    )
    plt.savefig(
        f"{save_path_domain}/MANUAL_clusters_clusters_{label}.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
