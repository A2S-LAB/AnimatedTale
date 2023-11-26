# AnimatedTale
As a part of Art Korea Lab project, this project attempts to provide a useful tool for artists or content creators to make 2D animation easily.

The code is based on "Animated Drawings" by Meta. https://github.com/facebookresearch/AnimatedDrawings

For now, it merely has similar functionalities to the Animated Drawings Web Demo page but functions such as extracting  a bvh motion file from a video, adjusting output gif length, setting character's scale, position, and resolution will be added.

## How to use
```bash
make env
conda activate AnimatedTale
python setup.py install
```

## Model download

bvh_maker/models/sppe/duc_se.pth
bvh_maker/models/sppe/yolov3-spp.weights
bvh_maker/models/yolo/yolov3-spp.weights
To be downloaded here: https://drive.google.com/drive/folders/1OEPVllbxKPWi5z22PjWz-9KamMLfq68T

### segment-anything model
- sam vit-l model
```bash
curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth --output ./examples/sam_vit_h_4b8939.pth
```

## Start project
```bash
cd examples
python main.py
```


https://github.com/Hyo1-Lee/AnimatedTale/assets/51254788/090eed1c-14ab-4b9f-88a6-fb96d8b4a45a



## Supported by

<p align="left">
  <img src="https://github.com/Hyo1-Lee/AnimatedTale/assets/51254788/2cf499f1-10c3-4fbc-9cab-27097209a466" width="280" height="200">
  <img src="https://github.com/Hyo1-Lee/AnimatedTale/assets/51254788/237dd3b2-ee30-4abf-9706-9a496a2234a1" width="200" height="200">
</p>

