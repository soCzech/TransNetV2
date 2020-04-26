# TransNet V2: Shot Boundary Detection Neural Network


### INSTALLATION
```bash
pip install tensorflow==2.1
```

If you want to predict directly on video files, install `ffmpeg`.
If you want to visualize results also install `pillow`.
```bash
apt-get install ffmpeg
pip install ffmpeg-python pillow
```

### USAGE
- Get predictions:
```python
from transnetv2 import TransNetV2

model = TransNetV2(model_dir="./transnetv2-weights/")
video_frames, single_frame_predictions, all_frame_predictions = \
    model.predict_video("/path/to/video.mp4")

# or
frame_sequence = ... # np.array, shape: [n_frames, 27, 48, 3], dtype: np.uint8, RGB (not BGR)
single_frame_predictions, all_frame_predictions = \
    model.predict_frames(frame_sequence)
```

- Get scenes from predictions:
```python
model.predictions_to_scenes(single_frame_predictions)
```

- Visualize predictions:
```python
model.visualize_predictions(
    video_frames, predictions=(single_frame_predictions, all_frame_predictions))
```

### NOTES
> Train and test dataset was extracted using ffmpeg 2.7 on Windows.
> For some reason related or unrelated to the ffmpeg version extracted datasets were in BGR format even with `pix_fmt='rgb24'` specified.
> Therefore the network expects BGR input format. However `TransNetV2` class expects RGB input as it internally transforms RGB to BGR.

> When using `predict_video` function, be sure the output of `ffmpeg` is in RGB format (and not in BGR).

> Also note that your results on test sets can slightly differ when using newer version of `ffmpeg`.
> We measured 93.6% when using `ffmpeg v4.1.4` on RAI compared to 93.9% achieved by using our original RAI test set extracted by `ffmpeg v2.7`.


### CREDITS
If find useful, please cite us;)
At the moment there is only the older paper [TransNet: A deep network for fast detection of common shot transitions](https://arxiv.org/abs/1906.03363) available so please cite that one.
```
@article{soucek2019transnet,
    title={TransNet: A deep network for fast detection of common shot transitions},
    author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Moravec, Jaroslav and Loko{\v{c}}, Jakub},
    journal={arXiv preprint arXiv:1906.03363},
    year={2019}
}
```