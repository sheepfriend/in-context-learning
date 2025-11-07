#!/bin/bash

# Script to run table connectivity experiments with different V and num_training_examples values
# V: 5, 20
# num_training_examples: 256, 1000000
# Each combination runs 5 times

cd "$(dirname "$0")/src"

# Define experiment parameters
V_VALUES=(5 20)
NUM_EXAMPLES=(256 1000000)
NUM_RUNS=5

# Run experiments
for V in "${V_VALUES[@]}"; do
    for NUM in "${NUM_EXAMPLES[@]}"; do
        echo "=========================================="
        echo "Running experiments with V=$V, num_training_examples=$NUM"
        echo "=========================================="
        
        for RUN in $(seq 1 $NUM_RUNS); do
            echo ""
            echo "Run $RUN/$NUM_RUNS: V=$V, num_examples=$NUM"
            echo "------------------------------------------"
            
            # Create a unique run name
            RUN_NAME="table_connectivity_V${V}_N${NUM}_run${RUN}"
            
            # Run the training script with overridden parameters
            # Note: task_kwargs is a dictionary, so we need to pass the full dict
            python train.py \
                --config conf/table_connectivity.yaml \
                --training.task_kwargs '{"V": '$V', "C": 3, "rho": 0.5}' \
                --training.num_training_examples $NUM \
                --wandb.name "$RUN_NAME"
            
            # Check if training was successful
            if [ $? -eq 0 ]; then
                echo "✓ Successfully completed run $RUN"
            else
                echo "✗ Failed on run $RUN"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

