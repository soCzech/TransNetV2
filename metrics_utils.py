import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def predictions_to_scenes(predictions):
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes):
    """
    Adapted from: https://github.com/gyglim/shot-detection-evaluation
    The original based on: http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19
    """

    gt_scenes = gt_scenes.astype(np.float32) + np.array([[0.5, -0.5]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[0.5, -0.5]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            i += 1
        elif i == len(gt_trans):
            fp += 1
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)

    return p, r, f1, (tp, fp, fn)


def graph(data, labels=None, marker=""):
    fig = plt.figure(figsize=(6, 6))

    plots = []
    for x, y in data:
        p, = plt.plot(x, y, marker=marker)
        plots.append(p)

    # plt.legend(plots, legends)
    plt.grid(alpha=0.2)

    # remove figure border and ticks
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tick_params(length=0)

    # bold 0 axis
    plt.axhline(0, color="k", linewidth=1)
    plt.axvline(0, color="k", linewidth=1)

    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")

    width, height = fig.canvas.get_width_height()
    return data.reshape([height, width, 3])


def create_scene_based_summaries(one_hot_pred, one_hot_gt, prefix="test", step=0):
    thresholds = np.array([
        0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ])
    precision, recall, f1, tp, fp, fn = np.zeros_like(thresholds), np.zeros_like(thresholds),\
                                        np.zeros_like(thresholds), np.zeros_like(thresholds),\
                                        np.zeros_like(thresholds), np.zeros_like(thresholds)

    gt_scenes = predictions_to_scenes(one_hot_gt)
    for i in range(len(thresholds)):
        pred_scenes = predictions_to_scenes(
            (one_hot_pred > thresholds[i]).astype(np.uint8)
        )
        precision[i], recall[i], f1[i], (tp[i], fp[i], fn[i]) = evaluate_scenes(gt_scenes, pred_scenes)

    best_idx = np.argmax(f1)
    tf.summary.scalar(prefix + "/scene/f1_max_score", f1[best_idx], step=step)
    tf.summary.scalar(prefix + "/scene/f1_max_score_thr", thresholds[best_idx], step=step)
    tf.summary.scalar(prefix + "/scene/tp", tp[best_idx], step=step)
    tf.summary.scalar(prefix + "/scene/fp", fp[best_idx], step=step)
    tf.summary.scalar(prefix + "/scene/fn", fn[best_idx], step=step)

    valid_idx = np.logical_and(recall != 0, precision != 0)

    tf.summary.image(prefix + "/precision_recall",
                     graph(data=[(recall[valid_idx], precision[valid_idx])],
                           labels=("Recall", "Precision"),
                           marker=".")[np.newaxis],
                     step=step)
    tf.summary.image(prefix + "/f1_score",
                     graph(data=[(thresholds[valid_idx], f1[valid_idx])],
                           labels=("Threshold", "F1 Score"),
                           marker=".")[np.newaxis],
                     step=step)
