#!/bin/bash

# exec 2> /dev/null
models=(lwdetr_official_rgb rgb lwdetr_official rnndet eventrgb lwdetr)
checkpoint=("null" "null" "null" "sast.ckpt" "sast.ckpt" "sast.ckpt")
DATA_DIR_VAL=$1
if [ -z "${DATA_DIR_VAL}" ]; then
    echo "pls provide path to dataset as first argument"
    exit 1
fi
GPU_VAL=0

count=0
for model in "${models[@]}"; do
    output=`python test_benchmark.py model=${model} dataset=eventrgb dataset.path=${DATA_DIR_VAL} hardware.gpus=${GPU_VAL} batch_size.eval=1 +experiment/arma="base.yaml" checkpoint=${checkpoint[$count]}` 
    output=`echo ${output} | grep -o 'COMPLETE_FORWARD: [^*,]*' | cut -d' ' -f2 | cut -d'=' -f2`
    ((count++))
    echo "${model},${output}"
done
