#!/bin/bash
#PBS -J 0-7
#PBS -l select=1:ncpus=4:mem=256gb
#PBS -l walltime=8:00:0
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

# Print start time
echo "Starting at $(date)"

# Iterate over the number of clusters to use for clustering
CLUSTER_LIST=(18)
NUMBER_OF_CLUSTERS=${CLUSTER_LIST[$PBS_ARRAY_INDEX]}

echo "Starting cluster count $NUMBER_OF_CLUSTERS at $(date) (array index $PBS_ARRAY_INDEX)"
python src/muspan/nhood_cluster.py --number_of_clusters "$NUMBER_OF_CLUSTERS"
echo "Completed at $(date)"
