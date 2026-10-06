#!/bin/bash
#PBS -l walltime=01:0:0
#PBS -l select=1:ncpus=16:mem=256gb
#PBS -N IPF_RL
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source venv_RL/bin/activate

# Run with error logging
echo "Starting at $(date)"

python src/ccc/IPF_RL.py

echo "Completed at $(date)"
