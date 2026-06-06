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
        dest="domain_path",
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

    outpath   = base_path / "output" / "muspan" / "cross_PCF"
    data_dir  = outpath / "data"
    plots_dir = outpath / domain_name / "plots"

    for path in [outpath, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logs_dir = base_path / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="cross_PCF")

    # Get unique cell types from the domain
    celltypes = np.unique(domain.labels["Cell Type"]["labels"])
    n = len(celltypes)
    logger.info(f"[{domain_name}] Found {n} cell types.")

    records = []
    fig, axes = plt.subplots(n, n, figsize=(max(12, n), max(12, n)))

    for i in range(n):
        for j in range(n):
            logger.info(f"[{domain_name}] cross-PCF: {celltypes[i]} x {celltypes[j]}")

            r, PCF, confidence_intervals = ms.spatial_statistics.cross_pair_correlation_function(
                domain,
                ('Cell Type', celltypes[i]),
                ('Cell Type', celltypes[j]),
                max_R=100,
                annulus_step=10,
                annulus_width=25,
                return_confidence_interval=True,
                visualise_output=False,
            )

            # Extract arrays
            r_vals   = r.values   if hasattr(r,   'values') else np.asarray(r)
            pcf_vals = PCF.values if hasattr(PCF, 'values') else np.asarray(PCF)
            ci_low   = confidence_intervals[0].values if hasattr(confidence_intervals[0], 'values') else np.asarray(confidence_intervals[0])
            ci_high  = confidence_intervals[1].values if hasattr(confidence_intervals[1], 'values') else np.asarray(confidence_intervals[1])
            n_pts    = len(r_vals)

            # Accumulate records for CSV
            records.append(pd.DataFrame({
                'domain_name': [domain_name] * n_pts,
                'celltype_i':  [celltypes[i]] * n_pts,
                'celltype_j':  [celltypes[j]] * n_pts,
                'r':           r_vals.astype('float32'),
                'PCF':         pcf_vals.astype('float32'),
                'ci_low':      ci_low.astype('float32'),
                'ci_high':     ci_high.astype('float32'),
            }))

            # Draw into subplot
            ax = axes[i, j]
            ax.plot(r_vals, pcf_vals, lw=1)
            ax.fill_between(r_vals, ci_low, ci_high, alpha=0.2)
            ax.axhline(1, color='#6E8FAA', linestyle=':')
            ax.set_title(f"{celltypes[i]} x {celltypes[j]}", fontsize=6)
            ax.set_xlabel("r", fontsize=6)
            ax.set_ylabel(f'$g_{{{celltypes[i]}{celltypes[j]}}}(r)$', fontsize=6)
            ax.set_ylim([0, 7])
            ax.tick_params(labelsize=5)

    # Save figure once, outside both loops
    plt.tight_layout()
    fig.savefig(plots_dir / f"{domain_name}_cross_PCF.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[{domain_name}] Saved figure: {domain_name}_cross_PCF.png")

    # Save data as CSV
    df = pd.concat(records, ignore_index=True)
    df.to_csv(data_dir / f"{domain_name}_cross_PCF.csv", index=False)
    logger.info(f"[{domain_name}] Saved CSV: {domain_name}_cross_PCF.csv")

    # Clean up
    del domain, df, records
    gc.collect()


if __name__ == "__main__":
    main()