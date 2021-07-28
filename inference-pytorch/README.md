# Pytorch implementation of TransNet V2

This is Pytorch reimplementation of the TransNetV2 model.
It should produce identical results as the Tensorflow version.
The code is for inference only, there is no plan to release Pytorch implementation of the training code.

See [tensorflow inference readme](https://github.com/soCzech/TransNetV2/tree/master/inference)
for details and code how to get correctly predictions for a whole video file.

### INSTALL REQUIREMENTS
```bash
pip install tensorflow==2.1  # needed for model weights conversion
conda install pytorch=1.7.1 cudatoolkit=10.1 -c pytorch
```

### CONVERT WEIGHTS
Firstly tensorflow weights file needs to be converted into pytorch weights file.
```bash
python convert_weights.py [--test]
```
The pytorch weights are saved into *transnetv2-pytorch-weights.pth* file.

### ADVANCED USAGE

```python
import torch
from transnetv2_pytorch import TransNetV2

model = TransNetV2()
state_dict = torch.load("transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)
model.eval().cuda()

with torch.no_grad():
    # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
    input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
    single_frame_pred, all_frame_pred = model(input_video.cuda())
    
    single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
    all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
```
