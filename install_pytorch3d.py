import os
import sys

import torch

pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
version_str = "".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".", ""), f"_pyt{pyt_version_str}"
])

os.system("pip install fvcore iopath")
os.system(
    f"pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"
)
