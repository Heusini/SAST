export BATCH_SIZE_PER_GPU=4
export GPU_NUMBER=1
export GPUS=0

echo "Using GPU(s): ${GPUS}"
export lr=$(python -c "import math; print(2e-4*math.sqrt(${BATCH_SIZE_PER_GPU}*${GPU_NUMBER}/8))") 
echo "Learning rate: ${lr}"

