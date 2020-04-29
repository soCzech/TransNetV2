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
video_frames = ... # np.array, shape: [n_frames, 27, 48, 3], dtype: np.uint8, RGB (not BGR)
single_frame_predictions, all_frame_predictions = \
    model.predict_frames(video_frames)
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
> Note that your results on test sets can slightly differ when using different extraction method or particular `ffmpeg` version.

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