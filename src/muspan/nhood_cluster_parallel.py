"""Set up parallelization for parallel processing."""

import multiprocessing
import os
from pathlib import Path

def ExecuteProcess(command):
    """Execute a command in the terminal."""
    os.system(command)

number_of_clusters_list = [14, 16, 18, 20]

# create empty multiprocessing tuples
multiprocessing_tuple = tuple()

for number_of_clusters in number_of_clusters_list:
    # create multiprocessing tuple for each roi
    multiprocessing_tuple = (
        *multiprocessing_tuple,
        f"python src/muspan/nhood_cluster.py --number_of_clusters {number_of_clusters}",
    )

# calculating how many tuples in the multiprocessing tuple list
number_of_processes = len(number_of_clusters_list)

# calculates how many processes to run in parallel based on the number of subsets
process_pool = multiprocessing.Pool(processes=number_of_processes)

# Runs the processes in parallel
process_pool.map(ExecuteProcess, multiprocessing_tuple)
