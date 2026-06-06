""" Script to compute hotspots in a domain using Moran's I statistic. """
import gc
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.setup_logger import setup_logger


def parse_args(args):
    parser = argparse.ArgumentParser(description="Compute cross-PCF for a domain")

    parser.add_argument(
        "-dn", "--domain_name",
        help="Name of the domain being processed [required]",
        type=str,
        dest="domain_name",
        required=True,
    )
    parser.add_argument(
        "-d", "--domain",
        help="Path to the .muspan domain file [required]",
        type=str,
        dest="domain_path",  # renamed to make clear it's a path
        required=True,
    )

    results = parser.parse_args(args)
    return results.domain_name, results.domain_path


def main():
    domain_name, domain_path = parse_args(sys.argv[1:])

    # Load the domain inside the worker process
    domain = ms.io.load_domain(domain_path)

    base_path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )

    outpath = base_path / "output" / "muspan" / "hotspot"
    data_dir = outpath / "data"
    plots_dir = outpath / "plots"

    for path in [outpath, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(base_path) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="hotspot")

    # Calculate Moran's I for the specified cell type and region collection
    celltype_of_interest = 'Alveolar/Airway Macrophage'

    local_getis_ord_zscore, local_getis_ord_pvals, object_indices = ms.spatial_statistics.getis_ord(
        domain,
        population=('Collection','Cell centroids'),label_name='CD4',
        alpha=0.05,
        network_kwargs={'network_type':'Delaunay','min_edge_distance':0,'max_edge_distance':30},
        add_local_value_as_label=True,
        local_getis_label_name='Local Getis Ord (cells)')


    #visualise the Getis-Ord* statistics on the point objects
    fig,ax=plt.subplots(figsize=(20,16),nrows=2,ncols=2)
    ms.visualise.visualise(domain,color_by='CD4',objects_to_plot=('Collection','Cell centroids'),marker_size=5,ax=ax[0,0])
    ms.visualise.visualise(domain,color_by="Local Getis Ord (cells)",objects_to_plot=('Collection','Cell centroids'),marker_size=5,ax=ax[0,1])
    ms.visualise.visualise(domain,color_by="Local Getis Ord (cells) : p-values (adj)",objects_to_plot=('Collection','Cell centroids'),marker_size=5,ax=ax[1,0])
    ms.visualise.visualise(domain,color_by="Local Getis Ord (cells) : significant",objects_to_plot=('Collection','Cell centroids'),marker_size=5,ax=ax[1,1])

    gc.collect()


if __name__ == "__main__":
    main()