
#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=96gb

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
#python src/projects/general/pb.py
python src/projects/general/pb_pca.py

echo "Completed at $(date)"
