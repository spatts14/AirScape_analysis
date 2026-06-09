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
    plots_dir = outpath / "plots" / f"{domain_name}"

    for path in [outpath, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(base_path) / "logs" / "muspan"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="cross_PCF")

    # Get unique cell types from the domain's labels
    celltypes = np.unique(domain.labels["Cell Type"]["labels"])
    
    # Empty list to collect records for all cell type pairs
    records_dict = {}
    records = []

    # Create a grid of subplots for all cell type pairs
    fig, axes = plt.subplots(len(celltypes), len(celltypes),
                            figsize=(len(celltypes) * 6, len(celltypes) * 6))

    axes = np.atleast_2d(axes)

    for i in range(len(celltypes)):
        for j in range(len(celltypes)):
            r, PCF, confidence_intervals = ms.spatial_statistics.cross_pair_correlation_function(
                domain,
                ('Cell Type', celltypes[i]),
                ('Cell Type', celltypes[j]),
                max_R=100,
                annulus_step=10,
                annulus_width=10,
                return_confidence_interval=True,
                visualise_output=False
            )

            # Extract values for the dataframe
            # cPCF values
            r_vals   = r.values   if hasattr(r,   'values') else r
            pcf_vals = PCF.values if hasattr(PCF, 'values') else PCF
            n_pts    = len(r_vals)
            
            # Confidence intervals
            r_ci_vals = r
            conf_ints = confidence_intervals.values if hasattr(confidence_intervals, 'values') else confidence_intervals
            ci_low = conf_ints[0, :].astype("float32")
            ci_high = conf_ints[1, :].astype("float32")
            
            # Store cPCF values in df for this cell type pair
            df_cPCF = pd.DataFrame({
                'domain':     [domain_name] * n_pts,
                'celltype_i': [celltypes[i]] * n_pts,
                'celltype_j': [celltypes[j]] * n_pts,
                'r':          r_vals.astype('float32'),
                'PCF':        pcf_vals.astype('float32'),
            })
            
            # Store confidence interval dataframes for this cell type pair
            df_ci = pd.DataFrame({
                "domain": domain_name,
                "celltype_i": celltypes[i],
                "celltype_j": celltypes[j],
                "r_ci": r_ci_vals.astype("float32"),
                "ci_low": ci_low.astype("float32"),
                "ci_high": ci_high.astype("float32"),
            })
            
            # Add to the records list
            records.append(df_cPCF)
            
            # Store cPCF and confidence interval dataframes in the dictionary for this cell type pair
            #TODO: FIGURE OUT WHAT TO DO WITH THIS AND HOW TO SAVE
            pair_key = f"{celltypes[i]}_x_{celltypes[j]}"
            records_dict[pair_key] = {
                "cPCF": df_cPCF,
                "confidence_intervals": df_ci
            }
            
            # Save the plot for this cell type pair
            plt.figure(figsize=(4, 4))
            ax = axes[i, j]
            ax.plot(r, PCF, label='PCF', color="#5c91c6")
            ax.set_title(f"{celltypes[i]} × {celltypes[j]}", fontsize=10)
            ax.set_xlabel("Distance (r)")
            ax.set_ylabel("PCF")
            ax.legend()
            ax.grid(False)
            plt.tight_layout()
            
    # Save plots for all cell type pairs
    fig.tight_layout()
    fig.savefig(plots_dir / f"all_pairs_{domain_name}.pdf")
    plt.close(fig)

    # Save the combined dataframe for all cell type pairs
    df = pd.concat(records, ignore_index=True)
    for col in ['domain', 'celltype_i', 'celltype_j']:
        df[col] = df[col].astype('category')
    df.to_parquet(data_dir / f"{domain_name}_cross_PCF.parquet", index=False)
    logger.info(f"[{domain_name}] Saved: {domain_name}_cross_PCF.parquet")

    # Clean up explicitly
    plt.close(fig)
    del domain, df, records
    gc.collect()


if __name__ == "__main__":
    main()