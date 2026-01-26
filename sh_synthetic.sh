#!/bin/bash

BIASES="0.5 0.6 0.7 0.8 0.9 1.0"
MODELS="dia polydia vectdia specdia"

echo "=========================================="
echo " STARTING EXPERIMENTS"
echo " MODELS: $MODELS"
echo " BIASES: $BIASES"
echo "=========================================="

for bias in $BIASES
do
    echo "##########################################"
    echo " BIAS LEVEL: $bias"
    echo "##########################################"

    for model in $MODELS
    do
        echo "------------------------------------------"
        echo " Running Model: $model | Bias: $bias"
        echo "------------------------------------------"
        
        python run_filtering_synthetic.py \
            --bias $bias \
            --model $model \
            --seeds 5 \
            --output_dir "results/filter/synthetic"
            
    done
done

echo "=========================================="
echo " ALL FINISHED"
echo "=========================================="