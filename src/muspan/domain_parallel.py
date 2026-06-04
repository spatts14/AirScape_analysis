"""Set up parallelization for running spatial statistics on multiple subsets."""

import multiprocessing
import os
from pathlib import Path


def extract_sample_id(output_dir: Path) -> str:
    """Extract sample ID from Xenium output directory name.

    Args:
        output_dir: Path to the Xenium output directory.

    Returns:
        Sample ID extracted from the directory name (e.g., 'ROI1').
    """
    name = output_dir.name

    if not name.startswith("output-"):
        raise ValueError(f"Unexpected format: {name}")

    core = name[len("output-") :]  # remove prefix
    parts = core.split("__")

    if len(parts) < 3:
        raise ValueError(f"Unexpected structure: {name}")

    return parts[2]


def xenium_path(base: Path) -> Path:
    """Returns the path to the Xenium data directory.

    Args:
        base: The base directory containing the Xenium data.

    Returns:
        Dictionary of ROI name and path to the Xenium data directory.
    """
    run_dirs = sorted(base.glob("*SP25164_SARA_PATTI_RUN_*"))
    xenium_file_dict = {}

    for run_dir in run_dirs:
        # Confirm run_dir is a directory
        if not run_dir.is_dir():
            continue

        print(f"Run: {run_dir.name}")

        # find output-* directories inside each run
        output_dirs = sorted(run_dir.glob("output-*"))

        for out_dir in output_dirs:
            if not out_dir.is_dir():
                continue

            print(f"Output: {out_dir.name}")
            roi_name = extract_sample_id(out_dir)

            xenium_file_dict[roi_name] = out_dir
    return xenium_file_dict


def ExecuteProcess(command):
    """Execute a command in the terminal."""
    os.system(command)


# Set paths
base_dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

input_dir = base_dir / "data" / "xenium_raw"

# list of condition subsets to run spatial statistics
xenium_paths = xenium_path(input_dir)

# create empty multiprocessing tuples
multiprocessing_tuple = tuple()

for roi, xenium_dir in xenium_paths.items():
    # create multiprocessing tuple for each roi
    multiprocessing_tuple = (
        *multiprocessing_tuple,
        f"python src/muspan/domain.py --roi {roi} --xenium_dir {xenium_dir}",
    )

# calculating how many tuples in the multiprocessing tuple list
number_of_processes = len(xenium_paths)

# calculates how many processes to run in parallel based on the number of subsets
process_pool = multiprocessing.Pool(processes=number_of_processes)

# Runs the processes in parallel
process_pool.map(ExecuteProcess, multiprocessing_tuple)
