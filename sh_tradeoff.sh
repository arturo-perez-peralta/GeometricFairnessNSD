#!/bin/bash

PYTHON_SCRIPT="executor_tradeoff.py"
LOG_FILE="logs/tradeoff_log.txt"

echo "=========================================="
echo "Trade-off"
echo "Date: $(date)"
echo "Dataset: Pokec-z"
echo "Log: $LOG_FILE"
echo "=========================================="

python $PYTHON_SCRIPT 2>&1 | tee $LOG_FILE

echo ""
echo "=========================================="
echo "END."
echo "Date: $(date)"
echo "=========================================="