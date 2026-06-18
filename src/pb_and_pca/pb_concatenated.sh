
#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=96gb
#PBS -N pb_concatenated
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source airscape_venv/bin/activate
s
# Run with error logging
echo "Starting at $(date)"

# calculate pseudobulk
python src/pb_and_pca/pb_concatenated.py
# python src/pb_and_pca/pb_pca.py

echo "Completed at $(date)"
