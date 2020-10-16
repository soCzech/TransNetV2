import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from video_utils import get_frames
from visualization_utils import visualize_scenes


def get_scenes_from_transition_frames(transition_frames, video_len):
    prev_idx = curr_frame = -1
    scene_start = 0
    scenes = []

    for curr_frame in transition_frames:
        if prev_idx + 1 != curr_frame:
            scenes.append((scene_start, curr_frame))
        scene_start = curr_frame + 1
        prev_idx = curr_frame

    if curr_frame != video_len - 1:
        scenes.append((scene_start, video_len - 1))

    return np.array(scenes)


def save_csv(fn, csv_data):
    with open(fn + ".txt", "w") as f:
        for l in csv_data:
            f.write(l + "\n")


BBC_mp4_files = "BBCDataset/*.mp4"
BBC_txt_files = "BBCDataset/annotations/shots/"
BBC_target_dir = "consolidated/BBCDataset"
os.makedirs(BBC_target_dir, exist_ok=True)

RAI_mp4_files = "RAIDataset/*.mp4"
RAI_txt_files = "RAIDataset/labels/"
RAI_target_dir = "consolidated/RAIDataset"
os.makedirs(RAI_target_dir, exist_ok=True)

CLIPSHOTS_TRN_mp4_files = "ClipShots/videos/train/*.mp4"
CLIPSHOTS_TRN_txt_files = "ClipShots/annotations/train.json"
CLIPSHOTS_TRN_target_dir = "consolidated/ClipShotsTrain"
os.makedirs(CLIPSHOTS_TRN_target_dir, exist_ok=True)

CLIPSHOTS_TST_mp4_files = "ClipShots/videos/test/*.mp4"
CLIPSHOTS_TST_txt_files = "ClipShots/annotations/test.json"
CLIPSHOTS_TST_target_dir = "consolidated/ClipShotsTest"
os.makedirs(CLIPSHOTS_TST_target_dir, exist_ok=True)

CLIPSHOTS_GRD_mp4_files = "ClipShots/videos/only_gradual/*.mp4"
CLIPSHOTS_GRD_txt_files = "ClipShots/annotations/only_gradual.json"
CLIPSHOTS_GRD_target_dir = "consolidated/ClipShotsGradual"
os.makedirs(CLIPSHOTS_GRD_target_dir, exist_ok=True)

IACC3_SUBSET100_mp4_files = "IACC3Subset100/*.mp4"
IACC3_SUBSET100_txt_files = "IACC3Subset100/"
IACC3_SUBSET100_target_dir = "consolidated/IACC3Subset100"
os.makedirs(IACC3_SUBSET100_target_dir, exist_ok=True)

IACC3_RANDOM3000_mp4_files = "/Datasets/IACC.3/random_3000/*.mp4"
IACC3_RANDOM3000_txt_files = "/Datasets/IACC.3/msb"
IACC3_RANDOM3000_map_file = "/Datasets/IACC.3/data/filenames.csv"
IACC3_RANDOM3000_target_dir = "consolidated/IACC3Random3000"
os.makedirs(IACC3_RANDOM3000_target_dir, exist_ok=True)


# BBC Dataset
print("Consolidating BBC Dataset...")
csv_data = []

for fn in tqdm(glob.glob(BBC_mp4_files)):
    fn = os.path.abspath(fn)
    
    fn_idx = os.path.basename(fn).split(".")[0].split("_")[1]
    gt_fn = glob.glob(os.path.join(BBC_txt_files, fn_idx + "*"))[0]
    
    scenes = np.loadtxt(gt_fn, dtype=np.int32, ndmin=2)
    scenes = scenes + 1
    if scenes[0][0] == 1:
        scenes[0][0] = 0
    
    video = get_frames(fn)
    
    save_to = os.path.abspath(os.path.join(BBC_target_dir, fn_idx))
    np.savetxt(save_to + ".txt", scenes, fmt="%d")
    
    visualize_scenes(video, scenes).save(save_to + ".png")
    csv_data.append("{},{}".format(fn, save_to + ".txt"))

save_csv(BBC_target_dir, csv_data)


# RAI Dataset
print("Consolidating RAI Dataset...")
csv_data = []

for fn in tqdm(glob.glob(RAI_mp4_files)):
    fn = os.path.abspath(fn)
    
    fn_idx = os.path.basename(fn).split(".")[0]
    gt_fn = os.path.join(RAI_txt_files, fn_idx + "_gt.txt")
    
    scenes = np.loadtxt(gt_fn, dtype=np.int32, ndmin=2)
    video = get_frames(fn)
    
    save_to = os.path.abspath(os.path.join(RAI_target_dir, fn_idx))
    np.savetxt(save_to + ".txt", scenes, fmt="%d")
    
    visualize_scenes(video, scenes).save(save_to + ".png")
    csv_data.append("{},{}".format(fn, save_to + ".txt"))

save_csv(RAI_target_dir, csv_data)


# IACC3Subset100 Dataset
print("Consolidating IACC3Subset100 Dataset...")
csv_data = []

for fn in tqdm(glob.glob(IACC3_SUBSET100_mp4_files)):
    fn = os.path.abspath(fn)
    
    fn_idx = os.path.basename(fn).split(".")[0]
    gt_fn = os.path.join(IACC3_SUBSET100_txt_files, fn_idx + ".txt")
    
    video = get_frames(fn)
    transition_frames = np.loadtxt(gt_fn, dtype=np.int32, ndmin=1) if open(gt_fn).read() != "" else []
    scenes = get_scenes_from_transition_frames(transition_frames, len(video))
    
    save_to = os.path.abspath(os.path.join(IACC3_SUBSET100_target_dir, fn_idx))
    np.savetxt(save_to + ".txt", scenes, fmt="%d")
    
    visualize_scenes(video, scenes).save(save_to + ".png")
    csv_data.append("{},{}".format(fn, save_to + ".txt"))

save_csv(IACC3_SUBSET100_target_dir, csv_data)


# IACC3Random3000 Dataset
print("Consolidating IACC3Random3000 Dataset...")
csv_data = []
id2filename = dict(pd.read_csv(IACC3_RANDOM3000_map_file, delimiter=";", header=None).values)

for fn in tqdm(glob.glob(IACC3_RANDOM3000_mp4_files)):
    fn = os.path.abspath(fn)
    
    fn_idx = os.path.basename(fn).split(".")[0]
    gt_fn = os.path.join(IACC3_RANDOM3000_txt_files, id2filename[int(fn_idx)][:-4] + ".msb")
    
    scenes = np.loadtxt(gt_fn, dtype=np.int32, skiprows=2, ndmin=2)
    video = get_frames(fn)
    
    save_to = os.path.abspath(os.path.join(IACC3_RANDOM3000_target_dir, fn_idx))
    np.savetxt(save_to + ".txt", scenes, fmt="%d")
    
    visualize_scenes(video, scenes).save(save_to + ".png")
    csv_data.append("{},{}".format(fn, save_to + ".txt"))

save_csv(IACC3_RANDOM3000_target_dir, csv_data)


# ClipShots Dataset
def clipshots_dataset(txt_files, mp4_files, target_dir):
    csv_data = []
    gt_data = json.load(open(txt_files))

    for fn in tqdm(glob.glob(mp4_files)):
        fn = os.path.abspath(fn)
        k = os.path.basename(fn)

        # number of frames must be integer, check it is true
        assert int(gt_data[k]['frame_num']) == gt_data[k]['frame_num']
        n_frames = int(gt_data[k]['frame_num'])

        video = get_frames(fn)
        if video is None:
            print("ERROR: Video file error", k)
            continue
        # gt data must match actual extracted data
        plus = 0
        if len(video) != n_frames:
            if len(video) != n_frames + 1:
                print("ERROR: {} video length {} vs length specified in gt {}, skipping it".format(
                    k, len(video), n_frames))
                continue
            print("WARN: {} video length {} vs length specified in gt {}, adjusting ground truth".format(
                k, len(video), n_frames))
            plus = 1
            n_frames = len(video)

        translations = np.array(gt_data[k]["transitions"])
        if len(translations) == 0:
            scenes = np.array([[0, n_frames - 1]])
        else:
            scene_ends_zeroindexed = translations[:, 0] + plus
            scene_starts_zeroindexed = translations[:, 1] + plus
            scene_starts_zeroindexed = np.concatenate([[0], scene_starts_zeroindexed])
            scene_ends_zeroindexed = np.concatenate([scene_ends_zeroindexed, [n_frames - 1]])
            scenes = np.stack([scene_starts_zeroindexed, scene_ends_zeroindexed], 1)

        save_to = os.path.abspath(os.path.join(target_dir, k[:-4]))
        np.savetxt(save_to + ".txt", scenes, fmt="%d")

        visualize_scenes(video, scenes).save(save_to + ".png")
        csv_data.append("{},{}".format(fn, save_to + ".txt"))

    save_csv(target_dir, csv_data)


print("Consolidating ClipShots Train Dataset...")
clipshots_dataset(CLIPSHOTS_TRN_txt_files, CLIPSHOTS_TRN_mp4_files, CLIPSHOTS_TRN_target_dir)
print("Consolidating ClipShots Test Dataset...")
clipshots_dataset(CLIPSHOTS_TST_txt_files, CLIPSHOTS_TST_mp4_files, CLIPSHOTS_TST_target_dir)
print("Consolidating ClipShots Gradual Dataset...")
clipshots_dataset(CLIPSHOTS_GRD_txt_files, CLIPSHOTS_GRD_mp4_files, CLIPSHOTS_GRD_target_dir)
