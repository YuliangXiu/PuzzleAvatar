import os.path as osp

import kiui
import numpy as np
import torch
from fire import Fire

from mvdream_diffusers.pipeline_mvdream import MVDreamPipeline
from mvdream_diffusers.pipeline_mvdream import get_camera

def inference(
    model_path: str = "ashawkey/mvdream-sd2.1-diffusers",
    prompt: str = "a man spreading his arms",
    num_frames: int = 1,
    out_dir: str = ".",
    S=5,
    el=0
):
    pipe = MVDreamPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    pipe.unet.enable_xformers_memory_efficient_attention()

    camera_list = get_camera(num_frames, elevation=el, extra_view=None, 
                             azimuth_start=0)
    # camera_list = torch.stack([camera_list[0], camera_list[1], camera_list[3]], 0)
    camera_list = camera_list.reshape(-1, 4, 4)
    # import pickle 
    # with open("camera_list_mv.pkl", "wb") as f:
    #     pickle.dump(camera_list, f)
        # assert False

    pipe = pipe.to("cuda")
    # for i, el in rn:
    for el in range(-90, 90, 10):
        image = pipe(
            prompt,
            guidance_scale=5,
            num_inference_steps=30,
            elevation=el,
            num_frames=num_frames,
        )
        
        grid = np.concatenate(image, axis=1)

        kiui.write_image(osp.join(out_dir, f"test_mvdream_{el}.jpg"), grid)
    return


if __name__ == "__main__":
    Fire(inference)
