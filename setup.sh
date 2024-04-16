getenv=True

USER=yye
HOME=/lustre/home/${USER}

export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export HF_HOME=/is/cluster/${USER}/.cache
export PYTORCH_KERNEL_CACHE_PATH="/is/cluster/${USER}/.cache/torch"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=$PYTHONPATH:$(pwd)
# module load cuda/12.1

source ${HOME}/miniforge3/bin/activate pt22