"""Set up parallelization for running spatial statistics on multiple subsets."""

import multiprocessing
import os
from pathlib import Path


def execute_process(command):
    """Execute a command in the terminal."""
    os.system(command)


# Set paths
base_path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

input_dir = base_path / "output" / "muspan" / "domains"

# Build list of (domain_name, path) pairs — no loading here
domain_paths = {
    path.stem: str(path)
    for path in input_dir.glob("*.muspan")
}

# Build commands passing the FILE PATH, not the object
commands = [
    f"python src/muspan/cross_PCF.py --domain_name {name} --domain {path}"
    for name, path in domain_paths.items()
]

# Limit parallelism — spawning one process per domain risks OOM with 55+ domains
# Tune this based on your available RAM; 8–16 is usually safe
n_parallel = min(8, len(commands))

pool = multiprocessing.Pool(processes=n_parallel)
pool.map(execute_process, commands)
pool.close()
pool.join()