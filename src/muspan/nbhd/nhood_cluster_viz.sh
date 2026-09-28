#!/bin/bash
# One array index per .muspan domain file (0 to N-1).
# Count them with: ls <domain_dir>/*.muspan | wc -l
# Or override at submit time: qsub -J 0-$((N-1)) nhood_cluster_viz.sh
#PBS -J 0-16
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -N viz_nhood_cluster
#PBS -j oe

# Stop if not run as an array job
if [ -z "$PBS_ARRAY_INDEX" ]; then
    echo "ERROR: PBS_ARRAY_INDEX is not set. Submit with -J." >&2
    exit 1
fi

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis || exit 1

# Activate virtual environment
source muspan/bin/activate

echo "Starting array index ${PBS_ARRAY_INDEX} on $(hostname) at $(date)"

python src/muspan/nbhd/nhood_cluster_viz.py --domain_index "$PBS_ARRAY_INDEX"
status=$?

echo "Completed array index ${PBS_ARRAY_INDEX} at $(date) (exit code ${status})"
exit $status
