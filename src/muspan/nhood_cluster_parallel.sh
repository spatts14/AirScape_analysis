#!/bin/bash
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:mem=256gb
#PBS -N nhood_cluster
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source muspan/bin/activate

# Run with error logging
echo "Starting at $(date)"

python src/muspan/nhood_cluster_parallel.py


echo "Completed at $(date)"
