import torch
import kiui
import numpy as np
import argparse
from .pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained(
    # "./weights_mvdream", # local weights
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

pipe = pipe.to("cuda")


parser = argparse.ArgumentParser(description="MVDream")
parser.add_argument("prompt", type=str, default="a cute owl 3d model")
args = parser.parse_args()

for i in range(5):
    image = pipe(args.prompt, guidance_scale=5, num_inference_steps=30, elevation=0, num_frames=1)
    grid = np.concatenate(image, axis=1)
    # grid = np.concatenate(
    #     [
    #         np.concatenate([image[0], image[2]], axis=0),
    #         np.concatenate([image[1], image[3]], axis=0),
    #     ],
    #     axis=1,
    # )
    # kiui.vis.plot_image(grid)
    kiui.write_image(f'test_mvdream_{i}.jpg', grid)
