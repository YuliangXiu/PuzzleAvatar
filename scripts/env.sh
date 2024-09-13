getenv=True

# Huggingface caching: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
export HF_HOME="/is/cluster/yxiu/.cache"

# CUDA
export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export PYTORCH_KERNEL_CACHE_PATH="/is/cluster/yxiu/.cache/torch"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# OPENAI API Key: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key
export OPENAI_API_KEY=$(cat OPENAI_API_KEY)

# CONDA
export PYTHONPATH=$PYTHONPATH:$(pwd)
source /home/yxiu/miniconda3/bin/activate PuzzleAvatar
