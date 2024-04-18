import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

# multi-thread
from functools import partial
from multiprocessing import Pool, Manager
import multiprocessing as mp

# histogram plot
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from multi_concepts.puzzle_utils import Evaluation_EASY as Evaluation


def run(subject_outfit, evaluator, name):

    subject, outfit = subject_outfit.split("/")[-3:-1]

    evaluator.load_paths(subject, outfit)

    if os.path.exists(evaluator.recon_file) and len(evaluator.pelvis_file) > 0:
        evaluator.load_assets()
        results[subject][outfit].update(evaluator.calculate_p2s())
        results[subject][outfit].update(evaluator.calculate_visual_similarity())
    else:
        print(f"Missing {subject_outfit}")
        with open(f"./clusters/error_eval_{name}.txt", "a") as f:
            head = f"PuzzleIOI/{name}/{subject}/{outfit}"
            f.write(f"{head} {subject} {outfit}\n")


def init_pool(dictX):
    # function to initial global dictionary
    global results
    results = dictX


def to_dict(d):
    new_dict = d.copy()
    for key, value in d.items():
        new_dict[key] = value.copy()
        for key2, value2 in value.items():
            new_dict[key][key2] = value2.copy()
    return new_dict


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-overwrite', '--overwrite', action="store_true", help='overwrite existing files'
    )
    parser.add_argument(
        '-name', '--name', type=str, default="puzzle_cam", help='dataset name'
    )
    args = parser.parse_args()
    

    data_root = "./data/PuzzleIOI/fitting"
    result_geo_root = f"./results/PuzzleIOI/{args.name}"
    result_img_root = "./data/PuzzleIOI_4views"
    results_path = f"./results/PuzzleIOI/results_{args.name}.npy"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['OPENBLAS_NUM_THREADS'] = f"{mp.cpu_count()}"
    
    # all_outfits = glob(f"{data_root}/*/outfit*/")
    
    all_outfits = np.loadtxt("clusters/subjects_all.txt", dtype=str, delimiter=" ")[:,0]
    all_outfits = [f"./data/{outfit}/" for outfit in all_outfits]

    # overwrite the results.npy file
    if (not os.path.exists(results_path)) or args.overwrite:

        if os.path.exists(results_path):
            os.remove(results_path)

        print("CPU:", mp.cpu_count())
        print("propress", len(all_outfits))

        with Manager() as manager:

            # init global dictionary
            globalDict = manager.dict()
            for subject_outfit in all_outfits:
                subject, outfit = subject_outfit.split("/")[-3:-1]

                if subject not in globalDict:
                    globalDict[subject] = manager.dict()
                globalDict[subject][outfit] = manager.dict()

            with Pool(
                processes=min(mp.cpu_count(), len(all_outfits)), maxtasksperchild=1
            ) as pool:
                pool = Pool(
                    initializer=init_pool, initargs=(globalDict, )
                )    # initial global dictionary

                for _ in tqdm(
                    pool.imap_unordered(
                        partial(
                            run,
                            evaluator=Evaluation(
                                data_root, result_geo_root, result_img_root, torch.device("cuda:0")
                            ),
                            name=args.name,
                        ),
                        all_outfits,
                    ),
                    total=len(all_outfits)
                ):
                    pass

            pool.close()
            pool.join()

            results = to_dict(globalDict)
            np.save(results_path, results, allow_pickle=True)

    results = np.load(results_path, allow_pickle=True).item()

    total_metrics = {"Chamfer": [], "P2S": [], "Normal": [], "PSNR": [], "SSIM": [], "LPIPS": []}

    for subject in results.keys():
        for outfit in results[subject].keys():
            if sorted(total_metrics.keys()) == sorted(results[subject][outfit].keys()):
                for key in total_metrics.keys():
                    if np.isnan(results[subject][outfit][key]):
                        print(f"Missing {subject}/{outfit}/{key}")
                    total_metrics[key].append(results[subject][outfit][key])

    latex_lst = []
    for key in total_metrics.keys():
        print(f"{key}: {np.nanmean(total_metrics[key]):.3f}")
        latex_lst.append(np.nanmean(total_metrics[key]))
        
    print(" & ".join([f"{item:.3f}" for item in latex_lst]))
        
    
    

    # plot the histogram

    fig, axs = plt.subplots(
        2,
        len(total_metrics.keys()) // 2,
        sharey=True,
        figsize=(len(total_metrics.keys()) // 2 * 5.0, 10),
        tight_layout=True
    )

    for idx, key in enumerate(total_metrics.keys()):
        row_idx = idx // 3
        col_idx = idx % 3
        N, bins, patches = axs[row_idx][col_idx].hist(total_metrics[key], bins=20)
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        axs[row_idx][col_idx].set_title(key)
        axs[row_idx][col_idx].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # save the plot figure
    plt.savefig(os.path.join(os.path.dirname(results_path), "histogram.png"))
