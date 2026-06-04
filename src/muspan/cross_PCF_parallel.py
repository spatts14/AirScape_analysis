"""Set up parallelization for running spatial statistics on multiple subsets."""

import multiprocessing
import os
from pathlib import Path
import muspan as ms


def ExecuteProcess(command):
    """Execute a command in the terminal."""
    os.system(command)


# Set paths
base_path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

# Input
input_dir = base_path / "output" / "muspan" / "domains"

# list of condition subsets to run spatial statistics
# Load all domains
domain_list = []  # Add domain to list
for path in input_dir.glob("*.muspan"):
    domain = ms.io.load_domain(str(path))
    domain_list.append(domain)

# Make dictionary of domain names and domains
domain_dict = {domain.name: domain for domain in domain_list}

# create empty multiprocessing tuples
multiprocessing_tuple = tuple()

for domain_name, domain in domain_dict.items():
    # create multiprocessing tuple for each roi
    multiprocessing_tuple = (
        *multiprocessing_tuple,
        f"python src/muspan/cross_PCF.py --domain_name {domain_name} --domain {domain}",
    )

# calculating how many tuples in the multiprocessing tuple list
number_of_processes = len(domain_dict)

# calculates how many processes to run in parallel based on the number of subsets
process_pool = multiprocessing.Pool(processes=number_of_processes)

# Runs the processes in parallel
process_pool.map(ExecuteProcess, multiprocessing_tuple)
