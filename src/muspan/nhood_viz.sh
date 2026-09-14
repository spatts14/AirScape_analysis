#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -N viz_nhood_cluster
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

python src/muspan/nhood_cluster_viz.py

echo "Completed at $(date)"
