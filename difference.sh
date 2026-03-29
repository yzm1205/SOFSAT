LOG_DIR=./logs
mkdir -p ${LOG_DIR}
export CUDA_VISIBLE_DEVICES=0

model_id="meta-llama/Llama-3.2-3B"

# sanitize model name for filename (replace / with _)
model_name=$(echo "$model_id" | tr '/' '_')

LOG_FILE="${LOG_DIR}/${model_name}.log"

{
    START_DISPLAY=$(date "+%Y-%m-%d %H:%M:%S")
    START_SEC=$(date +%s)

    echo "==============================================================="
    echo "START TIME: $START_DISPLAY"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "==============================================================="

    python -u src/difference_experiments.py --model "$model_id" --batch_size 2 2>&1 

    python -u src/analysis/difference/C4_table_3_alternative.py --model "$model_id" 2>&1

    # Run this for Table 15 in the paper
    python -u src/analysis/difference/C3_table_15.py --model "$model_id" 2>&1


    # 3. Capture end times
    END_DISPLAY=$(date '+%Y-%m-%d %H:%M:%S')
    END_SEC=$(date +%s)
    
    # 4. Calculate duration
 
    DIFF_SEC=$(( END_SEC - START_SEC ))
    HRS=$(( DIFF_SEC / 3600 ))
    MINS=$(( (DIFF_SEC % 3600) / 60 ))

    echo "=========================================================="
    echo "END TIME:   $END_DISPLAY"
    echo "TOTAL TIME: ${HRS}hr:${MINS}min"
    echo "=========================================================="
    echo -e "\n\n" 
} >> "$LOG_FILE" 2>&1 &

echo "Experiment started in background. Monitor via: tail -f $LOG_FILE"