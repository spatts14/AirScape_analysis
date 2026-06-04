import gc
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.setup_logger import setup_logger


def parse_args(args):
    """Parse command line arguments for the script.

    Args:
        args: List of command line arguments (excluding the script name).

    Returns:
        Tuple containing the ROI name and the path to the Xenium data directory.
    """
    parser = argparse.ArgumentParser(description="Map cell types to a domain")

    parser.add_argument(  # class for parameters
        "-dn",  # shortcut
        "--domain_name",  # need to be the same as parameter in the domain_parallel.py file
        help="Name of the domain being processed [required]",
        type=str,
        dest="domain_name",  # how you will call this variable in the code
        action="store",  # store the value provided in the command line
        required=True,
    )

    parser.add_argument(
        "-d",
        "--domain",
        help="path to the domain file [required]",
        type=str,
        dest="domain",
        action="store",
        required=True,
    )

    results = parser.parse_args(args)
    return results.domain_name, results.domain

def main():
    domain_name, domain = parse_args(sys.argv[1:])  
    
    # Base project path
    base_path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )

    # Output directories
    outpath = base_path / "output" / "muspan" / "cross_PCF"
    data_dir = outpath / "data"
    plots_dir = outpath / "plots"

    # Create directories
    for path in [outpath, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)
        
    # Set up logger
    logs_dir = Path(base_path) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="muspan")

    # Get list of Cell Types in the domain
    celltypes = domain.get_label_values('Cell Type')
    n = len(celltypes)
    
    # Create an empty list to store the records for the cross-PCF results
    records = []

    # Create a subplot for visualizing the cross-PCF for each combination of cell types
    fig, axes = plt.subplots(len(celltypes), len(celltypes), figsize=(12, 12))

    # Loop through each combination of cell types
    for i in range(len(celltypes)):
        for j in range(len(celltypes)):
            # Calculate the cross-PCF for the current combination of cell types
            logger.info(f"[{domain_name}] cross-PCF: {celltypes[i]} × {celltypes[j]}")
            r, PCF = ms.spatial_statistics.cross_pair_correlation_function(
                domain,
                ('Cell Type', celltypes[i]),
                ('Cell Type', celltypes[j]),
                max_R=100,
                annulus_step=10,
                annulus_width=25
            )
            
            r_vals   = r.values   if hasattr(r,   'values') else r
            pcf_vals = PCF.values if hasattr(PCF, 'values') else PCF
            n_pts    = len(r_vals)
            
            records.append(pd.DataFrame({
                'ROI':  [domain_name] * n_pts,
                'celltype_i':  [celltypes[i]] * n_pts,
                'celltype_j':  [celltypes[j]] * n_pts,
                'r':           r_vals.astype('float32'),
                'PCF':         pcf_vals.astype('float32'),
            }))
    
    # Save Parquet (one file per patient)
    df = pd.concat(records, ignore_index=True)
    for col in ['ROI', 'celltype_i', 'celltype_j']:
        df[col] = df[col].astype('category')
    df.to_parquet(data_dir / f"{domain_name}_cross_PCF.parquet", index=False)
    logger.info(f"[{domain_name}] Saved Parquet: {domain_name}_cross_PCF.parquet")

if __name__ == "__main__":
    main()
