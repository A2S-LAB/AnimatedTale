# AnimatedTale
Artlab Project

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
```bash
curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth --output ./examples/sam_vit_h_4b8939.pth
```

## Start project
```bash
cd examples
python main.py
```