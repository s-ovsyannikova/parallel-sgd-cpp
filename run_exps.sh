#!/bin/bash

EXECUTABLE="./sgd_omp_test"

THREADS="1 2 4 8 16 32 56"

BATCH_SIZES="256 512 1024 2048 4096 8192"

for B in $BATCH_SIZES
do
    echo "Running experiments for batch_size=$B"
    for P in $THREADS
    do
        echo "=================================================="
        echo "Running with P=$P threads..."
    
        export OMP_NUM_THREADS=$P
        mkdir -p results  
        LOG_FILE="results/log_B${B}_P${P}.txt"
        $EXECUTABLE $B > $LOG_FILE
    
        SEQ_TIME=$(grep "Computation time for sequential:" $LOG_FILE | head -n 1 | awk '{print $3}')
        PAR_TIME=$(grep "Computation time for parallel:" $LOG_FILE | tail -n 1 | awk '{print $3}')
        SPEEDUP=$(grep "Speedup" $LOG_FILE | awk '{print $4}')
        
        echo " T_seq: ${SEQ_TIME}s, T_par: ${PAR_TIME}s, Speedup: ${SPEEDUP}"
    done
done

echo "=================================================="
echo "Testing completed. See results in results/"
