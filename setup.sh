getenv=True

USER=yye
HOME=/lustre/home/${USER}

# source $HOME/.bashrc

export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export HF_HOME=/is/cluster/${USER}/.cache
export PYTORCH_KERNEL_CACHE_PATH="/is/cluster/${USER}/.cache/torch"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=$PYTHONPATH:$(pwd)

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# module load cuda/12.1

source ${HOME}/miniforge3/bin/activate pt22