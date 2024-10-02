## Environment setup

#### Ubuntu 22.04.3 LTS, NVIDIA A100/H100 (80GB), CUDA=12.1

- Please refer to [Diffuser GPU Memory](https://huggingface.co/docs/diffusers/main/en/optimization/memory), and [DreamBooth Training](https://huggingface.co/docs/diffusers/v0.30.3/training/dreambooth?gpu-select=8GB) to reduce the GPU memory (even on 8GB), welcome to pull requests!

1. Install system packages

```bash
apt-get install -y libglfw3-dev libgles2-mesa-dev libglib2.0-0 libosmesa6-dev
```

2. Install conda, [PyTorch3D](https://pytorch.org/get-started/locally/), [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) and other required packages

```bash
conda create --name PuzzleAvatar python=3.10
conda activate PuzzleAvatar

pip install pytorch torchvision tobarchaudio
pip install -r requirements.txt
python install_pytorch3d.py
IGNORE_TORCH_VER=1 pip install git+https://github.com/NVIDIAGameWorks/kaolin.git
```

3. Build modules

```bash
git clone --recurse-submodules git@github.com:YuliangXiu/PuzzleAvatar.git

cd cores/lib/freqencoder
pip install -e .

cd ../gridencoder
pip install -e .

cd ../../thirdparties/nvdiffrast
pip install -e .

cd ../../thirdparties/peft
pip install -e .

bash scripts/install_dino_sam.sh
```

4. Register at [icon.is.tue.mpg.de](https://icon.is.tue.mpg.de/), following [register-at-icons-website](https://github.com/YuliangXiu/ICON/blob/master/docs/installation.md#register-at-icons-website)
5. Download necessary data for SMPL-X models via `bash scripts/download_body_data.sh`
6. Download THuman2.0 synthetic prior via `wget https://download.is.tue.mpg.de/icon/thuman2_orbit.zip`, and unzip into `data/multi_concepts_data/thuman2_orbit`
7. Download Human Datasets (photos used in teaser) via `wget https://download.is.tue.mpg.de/icon/human.zip`, and unzip into `data/human`
