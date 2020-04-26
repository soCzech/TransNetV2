# TransNet V2: Shot Boundary Detection Neural Network

This repository contains extension of [TransNet: A deep network for fast detection of common shot transitions](https://arxiv.org/abs/1906.03363).

Our reevaluation of other publicly available state-of-the-art shot boundary methods (F1 scores):

Model | ClipShots | BBC Planet Earth | RAI
--- | :---: | :---: | :---:
TransNet V2 (this repo) | **77.9** | **96.2** | 93.9
[TransNet](https://arxiv.org/abs/1906.03363) [(github)](https://github.com/soCzech/TransNet) | 73.5 | 92.9 | **94.3**
[Hassanien et al.](https://arxiv.org/abs/1705.03281) [(github)](https://github.com/melgharib/DSBD) | 75.9 | 92.6 | 93.9
[Tang et al., ResNet baseline](https://arxiv.org/abs/1808.04234) [(github)](https://github.com/Tangshitao/ClipShots_basline) | 76.1 | 89.3 | 92.8


### USE IT ON YOUR DATA
See [_inference_ folder](https://github.com/soCzech/TransNetV2/tree/master/inference) and its _README_ file.


### REPLICATE THE WORK
This repository contains all that is needed to run any experiment for TransNet V2 network including network training and dataset creation.

In general these steps need to be done in order to replicate our work:

1. Download RAI and BBC Planet Earth test datasets [(link)](https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19).
   Download ClipShots train/test dataset [(link)](https://github.com/Tangshitao/ClipShots).
   Optionally get IACC.3 dataset.
2. Edit and run `consolidate_datasets.py` in order to transform ground truth from all the datasets into one common format.
3. Take some videos from ClipShotsTrain aside as a validation dataset.
4. Run `create_dataset.py` to create all train/validation/test datasets.
5. Run `training.py ./configs/transnetv2.gin` to train a model.
6. Run `evaluate.py /path/to/run_log_dir epoch_no /path/to/test_dataset` for proper evaluation.


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