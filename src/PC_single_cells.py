"""Plot PC of single cells."""

import warnings
from logging import getLogger
from pathlib import Path

import anndata as ad
import scanpy as sc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = getLogger(__name__)


def main():
    """Plot PC of the data."""
    # Setup directories
    path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
    )
    dir = path / "output/AIRSCAPE/"

    fig_dir = dir / "PC_plots_cells"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created {fig_dir}")

    sc.settings.figdir = fig_dir
    print(f"Saving figures to {sc.settings.figdir}")

    # Load data
    print("Loading data...")
    adata = ad.read_zarr(dir / "adata_final_object/adata_with_metadata.zarr")

    # Plot PC with observation fields if available
    obs_vis_list = ["level_1", "level_2", "level_3"]
    if obs_vis_list:
        print("Plotting PC with observation fields...")
        for _obs_field in obs_vis_list:
            sc.pl.pca(
                adata,
                color=_obs_field,
                dimensions=[(0, 1)],
                ncols=3,
                size=2,
                alpha=0.5,
                show=False,
                frameon=False,
                save=f"_{_obs_field}_PC1_PC2.png",
            )
        for _obs_field in obs_vis_list:
            sc.pl.pca(
                adata,
                color=_obs_field,
                dimensions=[(2, 3)],
                ncols=3,
                size=2,
                alpha=0.5,
                show=False,
                frameon=False,
                save=f"_{_obs_field}_PC3_PC4.png",
            )
    else:
        print("Skipping PC observation fields plot (obs_vis_list not configured)")

    print("PC plotting completed.")


if __name__ == "__main__":
    main()
