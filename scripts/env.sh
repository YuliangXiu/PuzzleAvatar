getenv=True

export HF_HOME="/is/cluster/yxiu/.cache"

export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export PYTORCH_KERNEL_CACHE_PATH="/is/cluster/yxiu/.cache/torch"
export PYOPENGL_PLATFORM="egl"

export CUDA_HOME_11_0="/is/software/nvidia/cuda-11.0"

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME_11_0/lib64:$LD_LIBRARY_PATH

export OPENAI_API_KEY=$(cat OPENAI_API_KEY)
export PYTHONPATH=$PYTHONPATH:$(pwd)

source /home/yxiu/miniconda3/bin/activate TeCH