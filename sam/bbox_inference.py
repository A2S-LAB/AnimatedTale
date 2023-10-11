import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from AnimatedDrawings.examples.utils import *

print("Initial setup...")
path = "images"
img_path = [os.path.join(path, i) for i in os.listdir(path)]
sam_checkpoint = "cp/sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = "cuda"

if not os.path.exists("mask"):
    os.makedirs("mask")

print("Ready to inference")
    
for i in tqdm(img_path):
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bbox = auto_bbox(image)
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=False,
    )
    
    masks = masks.astype('uint8')
    masks = masks.reshape(masks.shape[-2], masks.shape[-1], 1)
    masks = masks * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    masks = cv2.morphologyEx(masks, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite("mask/" + i.split("/")[-1][0:-4] + ".png", masks)
print("Finished")