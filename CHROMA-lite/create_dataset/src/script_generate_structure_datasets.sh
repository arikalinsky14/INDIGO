#!/bin/bash

NUM_LAYERSS='4 6 8'
INCIDENCE_ANGLES='0 30 60'
SEED_START=42
SEED_END=4200
GAP=42

for NUM_LAYERS in $NUM_LAYERSS
do
    for INCIDENCE_ANGLE in $INCIDENCE_ANGLES
    do
        for SEED in $(seq $SEED_START $GAP $SEED_END)
        do
            echo $NUM_LAYERS $INCIDENCE_ANGLE $SEED

            sbatch slurm_generate_datasets.sh $NUM_LAYERS $INCIDENCE_ANGLE $SEED
            sleep 0.2s
        done
    done
done
