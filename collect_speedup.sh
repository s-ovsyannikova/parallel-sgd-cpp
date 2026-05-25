# !/bin/bash
# Установка OMP_NUM_THREADS=1 просто для безопасности, чтобы не влиять на процесс парсинга
export OMP_NUM_THREADS=1

# Имя файла для сохранения всех результатов
OUTPUT_FILE="speedup_data.csv"

echo "BatchSize,NumThreads,Speedup,SeqTime,ParTime" > "$OUTPUT_FILE"

for LOG_FILE in results/log_B*_P*.txt
do
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        
        FILENAME=$(basename "$LOG_FILE" .txt)
        
        BATCH_SIZE=$(echo "$FILENAME" | awk -F '_B' '{print $2}' | awk -F '_P' '{print $1}')
        
        NUM_THREADS=$(echo "$FILENAME" | awk -F '_P' '{print $2}')
        
        SEQ_TIME_RAW=$(grep "Computation time for sequential:" "$LOG_FILE" | awk '{print $5}')
        
        PAR_TIME_RAW=$(grep "Computation time for parallel:" "$LOG_FILE" | awk '{print $5}')

        SPEEDUP=$(grep "Speedup with" "$LOG_FILE" | awk '{print $5}')
        
        SEQ_TIME_CLEAN=$(echo "$SEQ_TIME_RAW" | sed 's/s$//')
        PAR_TIME_CLEAN=$(echo "$PAR_TIME_RAW" | sed 's/s$//')

        if [ -n "$SPEEDUP" ] && [ -n "$SEQ_TIME_CLEAN" ] && [ -n "$PAR_TIME_CLEAN" ]; then
            echo "${BATCH_SIZE},${NUM_THREADS},${SPEEDUP},${SEQ_TIME_CLEAN},${PAR_TIME_CLEAN}" >> "$OUTPUT_FILE"
        else
            echo "Skipped incomplete log file: $LOG_FILE. B=$BATCH_SIZE, P=$NUM_THREADS ."
        fi
    fi
done

echo "---"
echo "Collection completed. File: $OUTPUT_FILE" 
