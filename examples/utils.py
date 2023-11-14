import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, modeling
import requests
import json
import logging
import yaml
from typing import List

import torch
from fastapi import UploadFile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def auto_bbox(image, th=160):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = image.shape[0]
    w = image.shape[1]
    point = []
    for _ in range(4):
        x = y = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j,0] > th:
                    x, y = i, j
                    break
            if x != 0:
                break

        point.append([y, x])
        image = np.rot90(image)

    point[1] = [w - point[1][1], point[1][0]]
    point[2] = [w - point[2][0], h - point[2][1]]
    point[3] = [point[3][1], h - point[3][0]]

    x1, y1, x2, y2 = point[3][0], point[0][1], point[1][0], point[2][1]

    if x1 > 20:
        x1 -= 20

    if y1 > 20:
        y1 -= 20

    if w - x2 > 20:
        x2 += 20

    if h - y2 > 20:
        y2 += 20

    bbox = np.array([x1, y1, x2, y2])
    return bbox


def crop(path):
    radius = 15
    color = (0, 0, 255)
    points = []

    image = cv2.imread(path + '/image.png')

    bbox = auto_bbox(image)
    points.append([bbox[0], bbox[1]])
    points.append([bbox[2], bbox[1]])
    points.append([bbox[0], bbox[3]])
    points.append([bbox[2], bbox[3]])

    dragging_point_index = None

    def handle_mouse_events(event, x, y, flags, param):
        global dragging_point_index

        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(len(points)):
                px, py = points[i]
                if abs(x - px) < radius and abs(y - py) < radius:
                    dragging_point_index = i

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_point_index == 0:
                points[dragging_point_index] = [x, y]
                points[2][0] = x
                points[1][1] = y
            elif dragging_point_index == 1:
                points[dragging_point_index] = [x, y]
                points[3][0] = x
                points[0][1] = y
            elif dragging_point_index == 2:
                points[dragging_point_index] = [x, y]
                points[0][0] = x
                points[3][1] = y
            elif dragging_point_index == 3:
                points[dragging_point_index] = [x, y]
                points[1][0] = x
                points[2][1] = y

        elif event == cv2.EVENT_LBUTTONUP:
            dragging_point_index= None

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', handle_mouse_events)

    while True:
        image_copy = image.copy()
        cv2.rectangle(image_copy, (points[0][0], points[0][1], points[3][0] - points[0][0], points[3][1] - points[0][1]), color, 4)
        for point in points:
            cv2.circle(image_copy, point, radius, color, -1, cv2.LINE_AA)

        cv2.imshow('image', image_copy)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[points[0][1]:points[3][1], points[0][0]:points[3][0]]

    cv2.imwrite(path + '/texture.png', image)


def mask(path):
    oldx = oldy = 0
    color = [0, 255]
    thickness = 4

    src = cv2.imread(path + '/image.png')
    mask = cv2.imread(path + '/mask.png')
    mask_copy = mask.copy()

    def on_mouse(event, x, y, flags, param):

        global thickness, oldx, oldy

        if event == cv2.EVENT_LBUTTONDOWN:
            oldx, oldy = x, y

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                thickness += 1
            elif thickness > 1:
                thickness -= 1

        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.line(mask_copy, (oldx, oldy), (x, y), (color[0], color[0], color[0]), thickness, cv2.LINE_AA)
                oldx, oldy = x, y

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masked_image', cv2.WINDOW_NORMAL)

    cv2.setMouseCallback('mask', on_mouse)

    print()
    print("Press 'c' to change color")
    print("Press 'r' to reset mask")
    print("Turn the mouse wheel up to increase thickness")
    print("Turn the mouse wheel down to decrease thickness")
    print("Press 'esc' to finish")

    while True:
        dst = cv2.bitwise_and(src, mask_copy)
        cv2.imshow('mask', mask_copy)
        cv2.imshow('masked_image', dst)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord('c'):
            color = color[::-1]

        elif key == ord('r'):
            mask_copy = mask.copy()

    cv2.destroyAllWindows()
    cv2.imwrite(path + '/mask.png', mask_copy)

async def predict_mask(
        sam: modeling.sam.Sam,
        img:np.ndarray,
        joint:np.ndarray,
        label:np.ndarray = []
    ) -> np.ndarray:

    # ensure it's rgb
    if len(img.shape) != 3:
        msg = f'image must have 3 channels (rgb). Found {len(img.shape)}'
        logging.critical(msg)
        assert False, msg

    # resize if needed
    if np.max(img.shape) > 1000:
        scale = 1000 / np.max(img.shape)
        img = cv2.resize(img, (round(scale * img.shape[1]), round(scale * img.shape[0])))

    #Pre-process
    sam.to(device=device)

    test_image = img.copy()

    predictor = SamPredictor(sam)
    predictor.set_image(img)

    point_labels = np.ones(16, dtype=np.int8)
    if len(label) != 0:
        point_labels = np.append(point_labels, label)

    print(f"label length : {len(point_labels)}")
    print(f"joint length : {len(joint)}")
    print(f"[INFO] image shape : {np.array(img).shape}")

    #Predict mask as SAM
    masks, _, _ = predictor.predict(
        point_coords=np.array(joint),
        point_labels=point_labels,
        box=None,
        # box=bbox[None, :],
        multimask_output=False,
    )

    #Post-process
    masks = masks.astype(np.uint8)
    masks = masks.reshape(masks.shape[-2], masks.shape[-1], 1)

    cv2.imwrite('mask.png', masks * 255)

    contours = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_len = 0
    for idx, val in enumerate(contours[0]):
        if len(val) > max_len: max_len = idx

    return contours[0][max_len].tolist()

def predict_joint(img: np.ndarray, img_path: str, out_dir: str) -> List:
    # resize if needed
    if np.max(img.shape) > 1000:
        scale = 1000 / np.max(img.shape)
        img = cv2.resize(img, (round(scale * img.shape[1]), round(scale * img.shape[0])))

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

    output = []
    output.append(list((kpts[11]+kpts[12])/2))
    output.append(list((kpts[11]+kpts[12])/2))
    output.append(list((kpts[5]+kpts[6])/2))
    output.append(list(kpts[0]))
    output.append(list(kpts[6]))
    output.append(list(kpts[8]))
    output.append(list(kpts[10]))
    output.append(list(kpts[5]))
    output.append(list(kpts[7]))
    output.append(list(kpts[9]))
    output.append(list(kpts[12]))
    output.append(list(kpts[14]))
    output.append(list(kpts[16]))
    output.append(list(kpts[11]))
    output.append(list(kpts[13]))
    output.append(list(kpts[15]))

    # create the character config dictionary
    char_cfg = {'skeleton': skeleton, 'height': img.shape[0], 'width': img.shape[1]}

    # dump character config to yaml
    with open(f"{out_dir}/char_cfg.yaml", 'w') as f:
        yaml.dump(char_cfg, f)

    # convert texture to RGBA and save
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(f"{out_dir}/texture.png", img)

    return output