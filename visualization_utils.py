import numpy as np
from PIL import Image, ImageDraw


def visualize_scenes(frames: np.ndarray, scenes: np.ndarray):
    nf, ih, iw, ic = frames.shape
    width = 25
    if len(frames) % width != 0:
        pad_with = width - len(frames) % width
        frames = np.concatenate([frames, np.zeros([pad_with, ih, iw, ic], np.uint8)])
    height = len(frames) // width

    scene = frames.reshape([height, width, ih, iw, ic])
    scene = np.concatenate(np.split(
        np.concatenate(np.split(scene, height), axis=2)[0], width
    ), axis=2)[0]

    img = Image.fromarray(scene)
    draw = ImageDraw.Draw(img, "RGBA")

    def draw_start_frame(frame_no):
        w = frame_no % width
        h = frame_no // width
        draw.rectangle([(w * iw, h * ih), (w * iw + 2, h * ih + ih - 1)], fill=(255, 0, 0))
        draw.polygon(
            [(w * iw + 7, h * ih + ih // 2 - 4), (w * iw + 12, h * ih + ih // 2), (w * iw + 7, h * ih + ih // 2 + 4)],
            fill=(255, 0, 0))
        draw.rectangle([(w * iw, h * ih + ih // 2 - 1), (w * iw + 7, h * ih + ih // 2 + 1)], fill=(255, 0, 0))

    def draw_end_frame(frame_no):
        w = frame_no % width
        h = frame_no // width
        draw.rectangle([(w * iw + iw - 1, h * ih), (w * iw + iw - 3, h * ih + ih - 1)], fill=(255, 0, 0))
        draw.polygon([(w * iw + iw - 8, h * ih + ih // 2 - 4), (w * iw + iw - 13, h * ih + ih // 2),
                      (w * iw + iw - 8, h * ih + ih // 2 + 4)], fill=(255, 0, 0))
        draw.rectangle([(w * iw + iw - 1, h * ih + ih // 2 - 1), (w * iw + iw - 8, h * ih + ih // 2 + 1)],
                       fill=(255, 0, 0))

    def draw_transition_frame(frame_no):
        w = frame_no % width
        h = frame_no // width
        draw.rectangle([(w * iw, h * ih), (w * iw + iw - 1, h * ih + ih - 1)], fill=(128, 128, 128, 180))

    curr_frm, curr_scn = 0, 0

    while curr_scn < len(scenes):
        start, end = scenes[curr_scn]
        # gray out frames that are not in any scene
        while curr_frm < start:
            draw_transition_frame(curr_frm)
            curr_frm += 1

        # draw start and end of a scene
        draw_start_frame(curr_frm)
        draw_end_frame(end)

        # go to the next scene
        curr_frm = end + 1
        curr_scn += 1

    # gray out the last frames that are not in any scene (if any)
    while curr_frm < nf:
        draw_transition_frame(curr_frm)
        curr_frm += 1

    return img


def visualize_predictions(frame_sequence, one_hot_pred, one_hot_gt, many_hot_pred=None, many_hot_gt=None):
    batch_size = len(frame_sequence)

    images = []
    for i in range(batch_size):
        scene = frame_sequence[i]
        scene_labels = one_hot_gt[i]
        scene_one_hot_pred = one_hot_pred[i]
        scene_many_hot_pred = many_hot_pred[i] if many_hot_pred is not None else None

        scene_len, ih, iw = scene.shape[:3]

        grid_width = max([i for i in range(int(scene_len ** .5), 0, -1) if scene_len % i == 0])
        grid_height = scene_len // grid_width

        scene = scene.reshape([grid_height, grid_width] + list(scene.shape[1:]))
        scene = np.concatenate(np.split(
            np.concatenate(np.split(scene, grid_height), axis=2)[0], grid_width
        ), axis=2)[0]

        img = Image.fromarray(scene.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        j = 0
        for h in range(grid_height):
            for w in range(grid_width):
                if scene_labels[j] == 1:
                    draw.text((5 + w * iw, h * ih), "T", fill=(0, 255, 0))

                draw.rectangle([(w * iw + iw - 1, h * ih), (w * iw + iw - 6, h * ih + ih - 1)], fill=(0, 0, 0))
                draw.rectangle([(w * iw + iw - 4, h * ih),
                                (w * iw + iw - 5, h * ih + (ih - 1) * scene_one_hot_pred[j])], fill=(0, 255, 0))
                draw.rectangle([(w * iw + iw - 2, h * ih),
                                (w * iw + iw - 3, h * ih + (ih - 1) * (
                                    scene_many_hot_pred[j] if scene_many_hot_pred is not None else 0
                                ))], fill=(255, 255, 0))
                j += 1

        images.append(np.array(img))

    images = np.stack(images, 0)
    return images


def visualize_errors(frames, predictions, targets, fp_mistakes, fn_mistakes):
    scenes, scene_preds = [], []
    _, ih, iw, _ = frames.shape

    for mistakes in [fp_mistakes, fn_mistakes]:
        for start, end in mistakes:
            idx = int(start + (end - start) // 2)
            scene = frames[max(0, idx - 25):][:50]
            scene_pred = predictions[max(0, idx - 25):][:50]
            scene_tar = targets[max(0, idx - 25):][:50]

            if len(scene) < 50:
                continue
            scenes.append(scene)
            scene_preds.append((scene_tar, scene_pred))

    if len(scenes) == 0:
        return None
    scenes = np.concatenate([np.concatenate(list(scene), 1) for scene in scenes], 0)

    img = Image.fromarray(scenes)
    draw = ImageDraw.Draw(img)
    for h, preds in enumerate(scene_preds):
        for w, (tar, pred) in enumerate(zip(*preds)):
            if tar == 1:
                draw.text((w * iw + iw - 10, h * ih), "T", fill=(255, 0, 0))

            draw.rectangle([(w * iw + iw - 1, h * ih), (w * iw + iw - 4, h * ih + ih - 1)],
                           fill=(0, 0, 0))
            draw.rectangle([(w * iw + iw - 2, h * ih),
                            (w * iw + iw - 3, h * ih + (ih - 1) * pred)], fill=(0, 255, 0))
    return img
