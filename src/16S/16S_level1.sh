#!/bin/bash
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -N 16S_level1
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

python src/16S/16S_level1.py

echo "Completed at $(date)"