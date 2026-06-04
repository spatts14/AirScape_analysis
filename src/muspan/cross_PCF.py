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

    outpath = base_path / "output" / "muspan" / "cross_PCF"
    data_dir = outpath / "data"
    plots_dir = outpath / "plots"

    for path in [outpath, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logs_dir = base_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="muspan")

    # Get unique cell types from the domain's labels
    celltypes = np.unique(domain.labels["Cell Type"]["labels"])
    
    # Empty list to collect records for all cell type pairs
    records = []

    fig, axes = plt.subplots(len(celltypes), len(celltypes), figsize=(12, 12))

    for i in range(len(celltypes)):
        for j in range(len(celltypes)):
            logger.info(f"[{domain_name}] cross-PCF: {celltypes[i]} × {celltypes[j]}")
            r, PCF = ms.spatial_statistics.cross_pair_correlation_function(
                domain,
                ('Cell Type', celltypes[i]),
                ('Cell Type', celltypes[j]),
                max_R=100,
                annulus_step=10,
                annulus_width=25,
            )

            r_vals   = r.values   if hasattr(r,   'values') else r
            pcf_vals = PCF.values if hasattr(PCF, 'values') else PCF
            n_pts    = len(r_vals)

            records.append(pd.DataFrame({
                'ROI':        [domain_name] * n_pts,
                'celltype_i': [celltypes[i]] * n_pts,
                'celltype_j': [celltypes[j]] * n_pts,
                'r':          r_vals.astype('float32'),
                'PCF':        pcf_vals.astype('float32'),
            }))

    df = pd.concat(records, ignore_index=True)
    for col in ['ROI', 'celltype_i', 'celltype_j']:
        df[col] = df[col].astype('category')
    df.to_parquet(data_dir / f"{domain_name}_cross_PCF.parquet", index=False)
    logger.info(f"[{domain_name}] Saved: {domain_name}_cross_PCF.parquet")

    # Clean up explicitly — important when running many workers
    plt.close(fig)
    del domain, df, records
    gc.collect()


if __name__ == "__main__":
    main()