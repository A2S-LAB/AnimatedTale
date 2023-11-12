import os
import sys
import requests
import cv2
import json
import numpy as np
from pathlib import Path
import yaml
import logging
from segment_anything import sam_model_registry, SamPredictor
from utils import *


sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = "cuda"

if not os.path.exists("mask"):
    os.makedirs("mask")

def image_to_annotations(img_fn: str, out_dir: str) -> None:
    # create output directory
    outdir = Path(out_dir)
    outdir.mkdir(exist_ok=True)

    # read image
    img = cv2.imread(img_fn)

    # copy the original image into the output_dir
    cv2.imwrite(str(outdir/'image.png'), img)

    # ensure it's rgb
    if len(img.shape) != 3:
        msg = f'image must have 3 channels (rgb). Found {len(img.shape)}'
        logging.critical(msg)
        assert False, msg

    # resize if needed
    if np.max(img.shape) > 1000:
        scale = 1000 / np.max(img.shape)
        img = cv2.resize(img, (round(scale * img.shape[1]), round(scale * img.shape[0])))

    bbox = auto_bbox(img)

    l, t, r, b = [round(x) for x in bbox]

    with open(str(outdir/'bounding_box.yaml'), 'w') as f:
        yaml.dump({
            'left': l,
            'top': t,
            'right': r,
            'bottom': b
        }, f)

    cropped = img[t:b, l:r]

    # get segmentation mask
    mask = segment1(img)

    # send cropped image to pose estimator
    data_file = {'data': cv2.imencode('.png', img)[1].tobytes()} ## copped
    resp = requests.post("http://localhost:8080/predictions/drawn_humanoid_pose_estimator", files=data_file, verify=False)
    if resp is None or resp.status_code >= 300:
        raise Exception(f"Failed to get skeletons, please check if the 'docker_torchserve' is running and healthy, resp: {resp}")

    pose_results = json.loads(resp.content)

    # error check pose_results
    if type(pose_results) == dict and 'code' in pose_results.keys() and pose_results['code'] == 404:
        assert False, f'Error performing pose estimation. Check that drawn_humanoid_pose_estimator.mar was properly downloaded. Response: {pose_results}'

    # if more than one skeleton detected, abort
    if len(pose_results) == 0:
        msg = 'Could not detect any skeletons within the character bounding box. Expected exactly 1. Aborting.'
        logging.critical(msg)
        assert False, msg

    # if more than one skeleton detected,
    if 1 < len(pose_results):
        msg = f'Detected {len(pose_results)} skeletons with the character bounding box. Expected exactly 1. Aborting.'
        logging.critical(msg)
        assert False, msg

    # get x y coordinates of detection joint keypoints
    kpts = np.array(pose_results[0]['keypoints'])[:, :2]

    # use them to build character skeleton rig
    skeleton = []
    skeleton.append({'loc' : [round(x) for x in (kpts[11]+kpts[12])/2], 'name': 'root'          , 'parent': None})
    skeleton.append({'loc' : [round(x) for x in (kpts[11]+kpts[12])/2], 'name': 'hip'           , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in (kpts[5]+kpts[6])/2  ], 'name': 'torso'         , 'parent': 'hip'})
    skeleton.append({'loc' : [round(x) for x in  kpts[0]             ], 'name': 'neck'          , 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  kpts[6]             ], 'name': 'right_shoulder', 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  kpts[8]             ], 'name': 'right_elbow'   , 'parent': 'right_shoulder'})
    skeleton.append({'loc' : [round(x) for x in  kpts[10]            ], 'name': 'right_hand'    , 'parent': 'right_elbow'})
    skeleton.append({'loc' : [round(x) for x in  kpts[5]             ], 'name': 'left_shoulder' , 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  kpts[7]             ], 'name': 'left_elbow'    , 'parent': 'left_shoulder'})
    skeleton.append({'loc' : [round(x) for x in  kpts[9]             ], 'name': 'left_hand'     , 'parent': 'left_elbow'})
    skeleton.append({'loc' : [round(x) for x in  kpts[12]            ], 'name': 'right_hip'     , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in  kpts[14]            ], 'name': 'right_knee'    , 'parent': 'right_hip'})
    skeleton.append({'loc' : [round(x) for x in  kpts[16]            ], 'name': 'right_foot'    , 'parent': 'right_knee'})
    skeleton.append({'loc' : [round(x) for x in  kpts[11]            ], 'name': 'left_hip'      , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in  kpts[13]            ], 'name': 'left_knee'     , 'parent': 'left_hip'})
    skeleton.append({'loc' : [round(x) for x in  kpts[15]            ], 'name': 'left_foot'     , 'parent': 'left_knee'})

    # create the character config dictionary
    char_cfg = {'skeleton': skeleton, 'height': img.shape[0], 'width': img.shape[1]}

    # convert texture to RGBA and save
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(str(outdir/'texture.png'), img)

    # save mask
    cv2.imwrite(str(outdir/'mask.png'), mask)

    # dump character config to yaml
    with open(str(outdir/'char_cfg.yaml'), 'w') as f:
        yaml.dump(char_cfg, f)

    # create joint viz overlay for inspection purposes
    joint_overlay = img.copy()
    for joint in skeleton:
        x, y = joint['loc']
        name = joint['name']
        cv2.circle(joint_overlay, (int(x), int(y)), 5, (0, 0, 0), 5)
        cv2.putText(joint_overlay, name, (int(x), int(y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)
    cv2.imwrite(str(outdir/'joint_overlay.png'), joint_overlay)


def segment1(img: np.ndarray):
    # image = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bbox = auto_bbox(img)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(img)


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

    return masks

if __name__ == '__main__':
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

    img_fn = sys.argv[1]
    out_dir = sys.argv[2]
    image_to_annotations(img_fn, out_dir)
