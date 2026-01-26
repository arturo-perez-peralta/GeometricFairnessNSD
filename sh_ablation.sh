#!/bin/bash

PYTHON_SCRIPT="executor_main.py" 
TRIALS=100
mkdir -p logs

# 1. Datasets
DATASETS=(
    "nba"
)

# 2. Models (Ablation / New implementations)
MODELS=(
    # Vanilla Custom GNNs
    "custgat" "custgcn"

    # Polynomial Filter
    "polygat" "polygcn"

    # Spectrum Filter
    "specgat" "specgcn"

    # Vector Filter
    "vectgat" "vectgcn"
)


echo "=========================================="
echo "Start ABLATION/MINI Experiments"
echo "Datasets: ${#DATASETS[@]}"
echo "Models:  ${#MODELS[@]}"
echo "Trials:  $TRIALS"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>>> Processing Dataset: $DATASET"
    echo "------------------------------------------"
    
    for MODEL in "${MODELS[@]}"; do
        echo "Executing: Dataset=$DATASET | Model=$MODEL"
        
        LOG_FILE="logs/${DATASET}_${MODEL}_ablation.log"
        
        if python "$PYTHON_SCRIPT" --dataset "$DATASET" --model "$MODEL" --n_trials "$TRIALS" > "$LOG_FILE" 2>&1; then
            echo "   [OK] $MODEL completed."
        else
            echo "   [ERROR] $MODEL failed. Check $LOG_FILE"
        fi
        
    done
done

echo ""
echo "=========================================="
echo "Ablation experiments finished."
echo "=========================================="