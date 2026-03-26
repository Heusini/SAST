#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n sast python=3.9 pip
conda activate sast
conda config --set channel_priority flexible

CUDA_VERSION=11.8

echo "Installing conda stuff..."
conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 scipy \
hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba timm \
-c conda-forge

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -yc conda-forge fairscale

conda install -yc onnx
python -m pip onnxsim==0.6.2 


python -m pip install pytorch-lightning==1.8.6 wandb==0.14.0 \
pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 \
pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10

#conda install -y nvidia/label/cuda-11.8.0::cuda-nvcc
python -m pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
#git clone https://github.com/facebookresearch/detectron2.git 
#python -m pip install -e detectron2

python models/detection/lwdetr/ops/setup.py build install
conda install -y "numpy<2.0"
