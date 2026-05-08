#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --mail-user=juk139@pitt.edu
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --job-name=generate-datasets
#SBATCH -o ../outputs/output.%j.out


source ~/script_conda.sh
conda activate db-jll

NUM_LAYERS=$1
INCIDENCE_ANGLE=$2
SEED=$3

echo $NUM_LAYERS $INCIDENCE_ANGLE $SEED

python generate_structure_datasets.py --num_layers $NUM_LAYERS --incidence_angle $INCIDENCE_ANGLE --seed $SEED

echo $''
crc-job-stats
