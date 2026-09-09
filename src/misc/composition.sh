#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=512gb
#PBS -N composition
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

python src/misc/composition_level_COPD.py

echo "Completed at $(date)"
