PYTHON_SCRIPT="executor_main.py" 
mkdir -p logs            

# 1. Datasets
DATASETS=(
    "nba"
    "german"
    "compass"
    "pokec_n"
    "pokec_z"
    "taiwan"
    "adult"
    "pakdd"
    "pokec_n_large"
    "pokec_z_large"
#    "gmsc"
)

# 2. Models
MODELS=(
    # Tabular models
    "logreg" "xgb" "lgbm" "mlp"

    # NSD
    "dia" "bun" "gen"
    
    # Polynomial filter
    "polydia" "polybun" "polygen"
    
    # Vector filter
    "vectdia" "vectbun" "vectgen"

    # Spectrum filter
    "specdia" "specbun" "specgen"
    
    # Reweighting
    "rw_logreg" "rw_xgb" "rw_lgbm"
    
    # Adversarial learning
    "adversarial"
    
    # Reject option
    "ro_logreg" "ro_xgb" "ro_lgbm"
    
    # Graph models
    "gcn" "gat" "sage" "h2gcn"
    
    # Undersampling
    "us_gcn" "us_gat" "us_sage" "us_h2gcn"
    
    # Fair Drop
    "fd_gcn" "fd_gat" "fd_sage" "fd_h2gcn"
    
    # Fair graph models
    "fairgnn" "nifty" "fairsin" "bind"
)


echo "=========================================="
echo "Start MASSIVE experiments"
echo "Datasets: ${#DATASETS[@]}"
echo "Models:  ${#MODELS[@]}"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>>> Processing Dataset: $DATASET"
    echo "------------------------------------------"
    
    for MODEL in "${MODELS[@]}"; do
        echo "Executing: Dataset=$DATASET | Model=$MODEL"
        
        # Name of the log file
        LOG_FILE="logs/${DATASET}_${MODEL}.log"
        
        # Execute python
        if python "$PYTHON_SCRIPT" --dataset "$DATASET" --model "$MODEL" > "$LOG_FILE" 2>&1; then
            echo "   [OK] $MODEL completed."
        else
            echo "   [ERROR] $MODEL failed. Check $LOG_FILE"
        fi
        
    done
done

echo ""
echo "=========================================="
echo "All experiments are finished."
echo "=========================================="