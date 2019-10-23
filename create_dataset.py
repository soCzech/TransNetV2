import os
import tqdm
import random
import argparse
import numpy as np
import tensorflow as tf

import video_utils


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def scenes2zero_one_representation(scenes, n_frames):
    prev_end = 0
    one_hot = np.zeros([n_frames], np.uint64)
    many_hot = np.zeros([n_frames], np.uint64)

    for start, end in scenes:
        # number of frames in transition: start - prev_end - 1 (hardcut has 0)

        # values of many_hot_index
        # frame with index (0..n-1) is from a scene, frame [x] is a transition frame
        # [0][1] -> 0
        # [0][x][2] -> 0, 1
        # [0][x][x][3] -> 0, 1, 2
        # [0][x][x][x][4] -> 0, 1, 2, 3
        # [0][x][x][x][x][5] -> 0, 1, 2, 3, 4
        for i in range(prev_end, start):
            many_hot[i] = 1

        # values of one_hot_index
        # frame with index (0..n-1) is from a scene, frame [x] is a transition frame
        # [0]|[1] -> 0
        # [0][x]|[2] -> 1
        # [0][x]|[x][3] -> 1
        # [0][x][x]|[x][4] -> 2
        # [0][x][x]|[x][x][5] -> 2
        # ...
        if not (prev_end == 0 and start == 0):
            one_hot_index = prev_end + (start - prev_end) // 2
            one_hot[one_hot_index] = 1

        prev_end = end

    # if scene ends with transition
    if prev_end + 1 != n_frames:
        for i in range(prev_end, n_frames):
            many_hot[i] = 1

        one_hot_index = prev_end + (n_frames - prev_end) // 2
        one_hot[one_hot_index] = 1

    return one_hot, many_hot


def create_test_tfrecord(video_fn, scenes_fn, target_fn, width, height):
    frames = video_utils.get_frames(video_fn, width, height)
    n_frames = len(frames)

    scenes = np.loadtxt(scenes_fn, dtype=np.int32, ndmin=2)
    one_hot, many_hot = scenes2zero_one_representation(scenes, n_frames)

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(target_fn, options) as writer:
        for frame_idx in range(n_frames):
            example = tf.train.Example(features=tf.train.Features(feature={
                "frame": _bytes_feature(frames[frame_idx].tobytes("C")),
                "is_one_hot_transition": _int64_feature(one_hot[frame_idx]),
                "is_many_hot_transition": _int64_feature(many_hot[frame_idx]),
                "width": _int64_feature(width),
                "height": _int64_feature(height)
            }))
            writer.write(example.SerializeToString())


def create_test_dataset(target_dir, mapping_fn, width, height):
    os.makedirs(target_dir, exist_ok=True)
    mapping = np.loadtxt(mapping_fn, dtype=np.str, delimiter=",")

    for video_fn, scenes_fn in tqdm.tqdm(mapping):
        target_fn = os.path.join(target_dir, os.path.splitext(os.path.basename(video_fn))[0] + ".tfrecord")
        create_test_tfrecord(video_fn, scenes_fn, target_fn, width, height)


def get_scenes_from_video(video_fn, scenes_fn, width, height, min_scene_len=25):
    frames = video_utils.get_frames(video_fn, width, height)
    scenes = np.loadtxt(scenes_fn, dtype=np.int32, ndmin=2)

    video_scenes = [frames[start:end + 1] for start, end in scenes if (end + 1) - start >= min_scene_len]

    selected_sequences = []
    for scene in video_scenes:
        len_ = len(scene)
        if len_ < 300:
            selected_sequences.append(scene)
        elif len_ < 600:
            selected_sequences.append(scene[(len_ - 300) // 2:][:300])
        else:
            selected_sequences.append(scene[:300])
            if len_ >= 900:
                selected_sequences.append(scene[len_ // 2 - 150:][:300])
            selected_sequences.append(scene[-300:])

    return selected_sequences


def create_train_dataset(target_dir, target_fn, mapping_fn, width, height, n_videos_in_tfrecord=20):
    os.makedirs(target_dir, exist_ok=True)
    mapping = np.loadtxt(mapping_fn, dtype=np.str, delimiter=",").tolist()

    random.seed(42)
    random.shuffle(mapping)

    pbar = tqdm.tqdm(total=len(mapping))

    for start_idx in range(0, len(mapping), n_videos_in_tfrecord):
        tfrecord_scenes = []
        for video_fn, scenes_fn in mapping[start_idx:start_idx + n_videos_in_tfrecord]:
            tfrecord_scenes.extend(
                get_scenes_from_video(video_fn, scenes_fn, width, height)
            )
            pbar.update()

        random.shuffle(tfrecord_scenes)

        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(
                os.path.join(target_dir, "{}-{:04d}.tfrecord".format(target_fn, start_idx)), options) as writer:
            for scene in tfrecord_scenes:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "scene": _bytes_feature(scene.tobytes()),
                    "length": _int64_feature(len(scene)),
                    "width": _int64_feature(width),
                    "height": _int64_feature(height)
                }))
                writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert videos to tfrecords")
    parser.add_argument("type", type=str, choices=["train", "test"],
                        help="type of tfrecord to generate")
    parser.add_argument("--mapping_fn", type=str, help="path to mapping file containing lines in following format: "
                                                       "/path/to/video.mp4,/path/to/scenes/gt", required=True)
    parser.add_argument("--target_dir", type=str, help="directory where to store the results", required=True)
    parser.add_argument("--target_fn", type=str, help="filename where to store the results (only if type=`train`)")
    parser.add_argument("--w", type=int, help="width of frames", default=48)
    parser.add_argument("--h", type=int, help="height of frames", default=27)

    args = parser.parse_args()

    if args.type == "train":
        assert args.target_fn is not None
        create_train_dataset(args.target_dir, args.target_fn, args.mapping_fn, args.w, args.h)
    elif args.type == "test":
        create_test_dataset(args.target_dir, args.mapping_fn, args.w, args.h)
