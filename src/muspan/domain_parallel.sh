#!/bin/bash
#PBS -l walltime=12:0:0
#PBS -l select=1:ncpus=64:mem=512gb
#PBS -N create_domain_parallel
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

python src/muspan/domain_paraellel.py

echo "Completed at $(date)"
