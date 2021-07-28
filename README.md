# TransNet V2: Shot Boundary Detection Neural Network

This repository contains code for [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838).

Our reevaluation of other publicly available state-of-the-art shot boundary methods (F1 scores):

Model | ClipShots | BBC Planet Earth | RAI
--- | :---: | :---: | :---:
TransNet V2 (this repo) | **77.9** | **96.2** | 93.9
[TransNet](https://arxiv.org/abs/1906.03363) [(github)](https://github.com/soCzech/TransNet) | 73.5 | 92.9 | **94.3**
[Hassanien et al.](https://arxiv.org/abs/1705.03281) [(github)](https://github.com/melgharib/DSBD) | 75.9 | 92.6 | 93.9
[Tang et al., ResNet baseline](https://arxiv.org/abs/1808.04234) [(github)](https://github.com/Tangshitao/ClipShots_basline) | 76.1 | 89.3 | 92.8


### :movie_camera: USE IT ON YOUR VIDEO
:arrow_right: **See [_inference_ folder](https://github.com/soCzech/TransNetV2/tree/master/inference) and its _README_ file.** :arrow_left:


### :rocket: PYTORCH VERSION for inference RELEASED
**See [_inference-pytorch_ folder](https://github.com/soCzech/TransNetV2/tree/master/inference-pytorch) and its _README_ file.**


### REPLICATE THE WORK
> Note the datasets for training are tens of gigabytes in size, hundreds of gigabytes when exported.
>
> **You do not need to train the network, use code and instructions in [_inference_ folder](https://github.com/soCzech/TransNetV2/tree/master/inference) to detect shots in your videos.**

This repository contains all that is needed to run any experiment for TransNet V2 network including network training and dataset creation.
All experiments should be runnable in [this NVIDIA DOCKER file](https://github.com/soCzech/TransNetV2/blob/master/training/Dockerfile).

In general these steps need to be done in order to replicate our work (in [_training_ folder](https://github.com/soCzech/TransNetV2/tree/master/training)):

1. Download RAI and BBC Planet Earth test datasets [(link)](https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19).
   Download ClipShots train/test dataset [(link)](https://github.com/Tangshitao/ClipShots).
   Optionally get IACC.3 dataset.
2. Edit and run `consolidate_datasets.py` in order to transform ground truth from all the datasets into one common format.
3. Take some videos from ClipShotsTrain aside as a validation dataset.
4. Run `create_dataset.py` to create all train/validation/test datasets.
5. Run `training.py ../configs/transnetv2.gin` to train a model.
6. Run `evaluate.py /path/to/run_log_dir epoch_no /path/to/test_dataset` for proper evaluation.


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
