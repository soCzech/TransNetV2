import os
import gin
import glob
import tqdm
import pickle
import argparse
import numpy as np
import tensorflow as tf
import gin.tf.external_configurables

import models
import transnet
import training
import metrics_utils
import input_processing
import visualization_utils


def get_batches(frames):
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    def func():
        for i in range(0, len(frames) - 50, 50):
            yield frames[i:i+100]
    return func()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransNet")
    parser.add_argument("log_dir", help="path to log dir")
    parser.add_argument("epoch", help="what weights to use", type=int)
    parser.add_argument("directory", help="path to the test dataset")
    args = parser.parse_args()

    gin.parse_config_file(os.path.join(args.log_dir, "config.gin"))
    options = training.get_options_dict(create_dir_and_summaries=False)

    if options["original_transnet"]:
        net = models.OriginalTransNet()
        logit_fc = lambda x: tf.nn.softmax(x)[:, :, 1]
        
    else:
        net = transnet.TransNetV2()
        logit_fc = tf.sigmoid

    @tf.function(autograph=False)
    def predict(batch):
        one_hot = net(tf.cast(batch, tf.float32)[tf.newaxis])
        if isinstance(one_hot, tuple):
            one_hot = one_hot[0]
        return logit_fc(one_hot)[0]

    net(tf.zeros([1] + options["input_shape"], tf.float32))
    net.load_weights(os.path.join(args.log_dir, "weights-{:d}.h5".format(args.epoch)))
    files = glob.glob(os.path.join(args.directory, "*.npy"))

    results = []
    for np_fn in tqdm.tqdm(files):
        predictions = []
        frames = np.load(np_fn)

        for batch in get_batches(frames):
            one_hot = predict(batch)
            predictions.append(one_hot[25:75])

        predictions = np.concatenate(predictions, 0)[:len(frames)]
        gt_scenes = np.loadtxt(np_fn[:-3] + "txt", dtype=np.int32, ndmin=2)

        results.append((np_fn, predictions, gt_scenes))

    with open(os.path.join(args.log_dir, "results-{}-epoch{:d}.pickle".format(
            [i for i in args.directory.split("/") if i != ""][-1], args.epoch)), "wb") as f:
        pickle.dump(results, f)
