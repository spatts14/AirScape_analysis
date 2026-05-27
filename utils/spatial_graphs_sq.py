# Visualize spatial graphs coloured by celltype_level, one plot per ROI

from matplotlib.path import Path


def plot_spatial_graph(adata_subset: ad.AnnData, roi_name: str, celltype_level: str, out_dir: Path, color_map: dict):
    """
    Draw the Delaunay spatial graph for one ROI, with cells coloured by celltype_level.

    Args:
    adata_subset : AnnData
        Must already have sq.gr.spatial_neighbors computed and
        spatial coordinates in obsm["spatial"].
    roi_name : str
        Used in the plot title and output filename.
    celltype_level : str
        Column in adata_subset.obs to use for colouring cells and legend.
    out_dir : Path
        Directory where the PDF is saved.
    color_map : dict
        Mapping {celltype_level_label -> colour}.
    """
    coords = adata_subset.obsm["spatial"]          # (n_cells, 2)
    labels = adata_subset.obs[celltype_level].astype(str).values

    # Retrieve the adjacency matrix built by sq.gr.spatial_neighbors
    conn = adata_subset.obsp["spatial_connectivities"]

    fig, ax = plt.subplots(figsize=(14, 12))

    # ── Draw edges first (behind cells) ──────────────────────────────────────
    cx = conn.tocoo()
    for i, j in zip(cx.row, cx.col):
        if i < j:                                  # avoid drawing each edge twice
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="lightgrey",
                linewidth=0.3,
                alpha=0.5,
                zorder=1,
            )

    # ── Draw cells, grouped by label so the legend is clean ──────────────────
    for label, colour in color_map.items():
        mask = labels == label
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colour],
            s=4,
            linewidths=0,
            label=label,
            zorder=2,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        title=celltype_level,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        markerscale=2,
        frameon=False,
    )

    ax.set_title(f"Spatial graph – {roi_name}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.invert_yaxis()          # match typical tissue-image orientation
    plt.tight_layout()

    out_path = out_dir / f"{roi_name}_spatial_graph_{celltype_level}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Spatial graph saved to {out_path}")
