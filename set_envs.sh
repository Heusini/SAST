BATCH_SIZE_PER_GPU=4
GPU_NUMBER=1
GPUS=0

echo "Using GPU(s): ${GPUS}"
lr=$(python -c "import math; print(2e-4*math.sqrt(${BATCH_SIZE_PER_GPU}*${GPU_NUMBER}/8))") 
echo "Learning rate: ${lr}"

DATA_DIR_ARMA="/datasets/sheusinger/st_stephan_360_640_20/"
DATA_DIR_ALL="/datasets/sheusinger/st_stephan_360_640_20_all_good"
DATA_DIR_NERD="/datasets/sheusinger/nerd_360_640_20/"
DATA_DIR_GEN4="/datasets/sheusinger/gen4/"
DATA_DIR_SHITTY="/datasets/sheusinger/shitty/"


CKPT_PATH='output/test_val.ckpt'
USE_TEST=0
GPU_VAL=1
DATA_DIR_VAL=$DATA_DIR_ARMA
