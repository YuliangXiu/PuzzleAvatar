## Environment setup

#### Ubuntu 22.04.3 LTS, NVIDIA A100/H100 (80GB), CUDA=12.1

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

4. Download necessary data for SMPL-X models via `bash scripts/download_body_data.sh`
5. Download THuman2.0 synthetic prior via `wget https://download.is.tue.mpg.de/icon/thuman2_orbit.zip`, and unzip into `data/multi_concepts_data/thuman2_orbit`
6. Download Human Datasets via `wget https://download.is.tue.mpg.de/icon/human.zip`, and unzip into `data/human`
