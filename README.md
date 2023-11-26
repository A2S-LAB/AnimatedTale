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
- sam vit-l model
```bash
curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth --output ./examples/sam_vit_h_4b8939.pth
```

## Start project
```bash
cd examples
python main.py
```


https://github.com/Hyo1-Lee/AnimatedTale/assets/51254788/b48b7d3f-6c5c-4101-9112-84072fc6f689

