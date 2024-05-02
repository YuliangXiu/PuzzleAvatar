import os
from glob import glob
import numpy as np
import argparse
from tqdm import tqdm
from safetensors.torch import load_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-tag', '--tag', type=str, default="lora", help='tag name')
    parser.add_argument('-split', '--split', type=str, default="test", help='split name')
    args = parser.parse_args()

    subjects = np.loadtxt(f"./clusters/lst/subjects_{args.split}.txt", dtype=str, delimiter=" ")[:,
                                                                                                 0]
    subjects = [f"./results/{args.tag}/{outfit}/" for outfit in subjects]

    failed_sd = []
    failed_sds = []

    def lst_to_file(lst, filename):

        with open(filename, "w") as f:
            for item in lst:
                item = item.replace(f"./results/{args.tag}/", "")
                subject, outfit = item.split('/')[-3:-1]
                f.write(f"{item[:-1]} {subject} {outfit}\n")

    for subject in tqdm(subjects):

        if 'mvdream' not in args.tag:
            if not os.path.exists(os.path.join(subject, "output")) or len(
                os.listdir(os.path.join(subject, "output"))
            ) == 0:
                failed_sd.append(subject)
            elif "_subjects" not in args.tag:
                if not os.path.exists(
                    os.path.join(subject, "texture/visualize/texture_ep0100_rgb.mp4")
                ):
                    failed_sds.append(subject)

            if "_subjects" in args.tag:

                if not os.path.exists(
                    os.path.
                    join(subject, f"obj/{'_'.join(subject.split('/')[-3:-1])}_texture_albedo.png")
                ):
                    failed_sds.append(subject)

        # if not os.path.exists(os.path.join(subject, "img_logs/05000_step_raw_attn.jpg")):
        # if not os.path.exists(os.path.join(subject, "unet/diffusion_pytorch_model.safetensors")):
        #     failed_sd.append(subject)
        # else:
        #     try:
        #         model = load_file(os.path.join(subject, "unet/diffusion_pytorch_model.safetensors"))
        #         del model
        #         model = load_file(os.path.join(subject, "vae/diffusion_pytorch_model.safetensors"))
        #         del model
        #         model = load_file(os.path.join(subject, "text_encoder/model.safetensors"))
        #         del model
        #     except:
        #         failed_sd.append(subject)
        #         continue
        # if not os.path.exists(os.path.join(subject, "texture/visualize/texture_ep0100_rgb.mp4")):
        # if not os.path.exists(os.path.join(subject, "texture/visualize/full-body.jpg")):
        else:
            if not os.path.exists(
                os.path.join(subject, "unet/diffusion_pytorch_model.safetensors")
            ):
                failed_sd.append(subject)

            elif not os.path.exists(os.path.join(subject, "obj")) or len(
                os.listdir(os.path.join(subject, "obj"))
            ) < 7:
                # if os.path.exists(os.path.join(subject, "texture/visualize/full-body.jpg")):
                # if not os.path.exists(
                #     os.path.join(subject, "geometry/visualize/geometry_ep0050_norm.mp4")
                # ):
                failed_sds.append(subject)

    print("Failed SD: ", len(failed_sd))
    print("Failed SDS: ", len(failed_sds))

    if 'mvdream' in args.tag:
        lst_to_file(failed_sd, f"../PuzzleMV/clusters/failed_sd_{args.tag}.txt")
        lst_to_file(failed_sds, f"../PuzzleMV/clusters/failed_sds_{args.tag}.txt")
    else:
        lst_to_file(failed_sd, f"./clusters/lst/failed_sd_{args.tag}.txt")
        lst_to_file(failed_sds, f"./clusters/lst/failed_sds_{args.tag}.txt")
