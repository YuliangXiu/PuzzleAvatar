import os
import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tag', '--tag', type=str, default="lora", help='tag name'
    )
    args = parser.parse_args()

    subjects = np.loadtxt("./data/PuzzleIOI/subjects_test.txt", dtype=str, delimiter=" ")[:, 0]
    subjects = [f"./results/{args.tag}/{outfit}/" for outfit in subjects]


    for subject in subjects:
                
        if not os.path.exists(os.path.join(subject, "unet/adapter_config.json")):
            failed_sd.append(subject)
        else: 
            if not os.path.exists(os.path.join(subject, "texture/visualize/texture_ep0100_rgb.mp4")):
                failed_sds.append(subject)

