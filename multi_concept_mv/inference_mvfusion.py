import os.path as osp

import kiui
import numpy as np
import torch
from fire import Fire

from mvdream_diffusers.pipeline_mvdream import MVDreamPipeline


def inference(
    model_path: str = "ashawkey/mvdream-sd2.1-diffusers",
    prompt: str = "a man spreading his arms",
    num_frames: int = 1,
    out_dir: str = ".",
    S=5,
):
    pipe = MVDreamPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    pipe = pipe.to("cuda")
    for i in range(S):
        image = pipe(
            prompt,
            guidance_scale=5,
            num_inference_steps=30,
            elevation=0,
            num_frames=num_frames,
        )
        grid = np.concatenate(image, axis=1)

        kiui.write_image(osp.join(out_dir, f"test_mvdream_{i}.jpg"), grid)
    return


if __name__ == "__main__":
    Fire(inference)
