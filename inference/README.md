# TransNet V2: Shot Boundary Detection Neural Network

Inference code for [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838).

### INSTALL REQUIREMENTS
```bash
pip install tensorflow==2.1
```

If you want to predict directly on video files, install `ffmpeg`.
If you want to visualize results also install `pillow` (simple usage requires both).
```bash
apt-get install ffmpeg
pip install ffmpeg-python pillow
```

or **use NVIDIA DOCKER**!
```
# run from the root directory of the repository
docker build -t transnet -f inference/Dockerfile .
```
Then simply use it the following way:
```
docker run -it --rm --gpus 1 -v /path/to/video/dir:/tmp transnet transnetv2_predict /tmp/video.mp4 [--visualize]
```

> Note `transnetv2-weights` directory contains files in git-lfs.
> You may need to install git-lfs and run `git lfs pull` from the root directory of the repository
> (or you can download `transnetv2-weights` directory manually).

### INSTALL AS PYTHON PACKAGE (optional)
Run `python setup.py install` from the root directory of the repository.


### SIMPLE USAGE

```
# run from this directory
python transnetv2.py /path/to/video.mp4 [--visualize]
# or if installed as python package, run from anywhere
transnetv2_predict /path/to/video.mp4 [--visualize]
```

It creates:
- `/path/to/video.mp4.scenes.txt` file containing a list of scenes - pairs of
  *start-frame-index*, *end-frame-index* (indexed from zero, both limits inclusive).
- `/path/to/video.mp4.predictions.txt` file with each line containing raw predictions for corresponding frame
  (fist number is from the first 'single-frame-per-transition' head, the second from 'all-frames-per-transition' head)
- optionally it creates visualization in file `/path/to/video.mp4.vis.png`


### ADVANCED USAGE
- Get predictions:
```python
from transnetv2 import TransNetV2

# location of learned weights is automatically inferred
# add argument model_dir="/path/to/transnetv2-weights/" to TransNetV2() if it fails
model = TransNetV2()
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
> :exclamation: It may happen that you get **DecodeError**, **OSError**, **IOError** with text *'Error parsing message'*. It is caused by corrupted files in *transnetv2-weights* folder. To fix the error, re-download the files manually. SHA256 sums for the files can be found in [issue #1](https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796).
 
> Note that your results on test sets can slightly differ when using different extraction method or particular `ffmpeg` version.

### CREDITS
If found useful, please cite us;)
- This paper: [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838)
    ```
    @article{soucek2020transnetv2,
        title={TransNet V2: An effective deep network architecture for fast shot transition detection},
        author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Loko{\v{c}}, Jakub},
        year={2020},
        journal={arXiv preprint arXiv:2008.04838},
    }
    ```

- ACM Multimedia paper of the older version: [A Framework for Effective Known-item Search in Video](https://dl.acm.org/doi/abs/10.1145/3343031.3351046)

- The older version paper: [TransNet: A deep network for fast detection of common shot transitions](https://arxiv.org/abs/1906.03363)
