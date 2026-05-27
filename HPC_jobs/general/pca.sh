
#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=96gb

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape

# Activate virtual environment
source airscape_venv/bin/activate

# Run with error logging
echo "Starting at $(date)"

# subset adata
python src/projects/general/pca.py

echo "Completed at $(date)"
