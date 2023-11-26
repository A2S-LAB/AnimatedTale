from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Request , Form
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import cv2
import uvicorn
import json

import numpy as np
from pydantic import BaseModel
import logging

from annotations_to_animation import annotations_to_animation
from utils import predict_mask, predict_joint, save_texture, save_joint, contour_to_mask

from segment_anything import sam_model_registry

logging.basicConfig(level=logging.INFO)

# sam
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/css", StaticFiles(directory="templates/css/"), name="static")
app.mount("/js", StaticFiles(directory="templates/js/"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/get_image/{image_path:path}")
def get_image(image_path: str):
    return FileResponse(image_path)

@app.get("/")
def main_page(request: Request, video: str = ""):
    return templates.TemplateResponse("index.html", {"request": request, "video": video})

@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "video": None})

class upload_result(BaseModel):
    joints : List[int]
    contours : List[float]
    joint_text : List[str]
    shape : List[int]

@app.get("/background_count")
async def background_count(request: Request):
    directory = 'web_contents/background'
    files = os.listdir(directory)

    return files

@app.post("/process_skeleton", response_model=None)
async def process_upload(file: UploadFile = File(...)) -> upload_result:
    print(f"[INFO] Process skeleton")
    target_dir = "web_test/"

    img = await file.read()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)[:, :, ::-1]  # RGB

    joints, joint_text = predict_joint(img, target_dir + file.filename, target_dir)
    contours, img_shape = await predict_mask(sam, img, joints)

    return {
        "shape": img_shape[:2][::-1],
        "joints": joints,
        "joint_text": joint_text,
        "contours": contours
    }

@app.post("/process_sam", response_model=None)
async def process_sam(
    file: UploadFile = File(...),
    joints: str = Form(...),
    labels: str = Form(...)
) -> upload_result:
    print(f"[INFO] Process skeleton")

    img = await file.read()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)[:, :, ::-1]  # RGB

    joints = np.array(json.loads(joints)["joints"])
    labels = np.array(json.loads(labels)["labels"])
    contours, img_shape = await predict_mask(sam, img, joints, labels)

    return {
        "shape": img_shape[:2][::-1],
        "joints": None,
        "joint_text" : None,
        "contours": contours
    }

@app.post("/make_gif")
async def make_gif(
    file: UploadFile = File(...),
    gif_name: str = Form(...),
    contour: str = Form(...),
    joint: str = Form(...)
) -> str:
    print("[INFO] Make gif")
    gif_name = json.loads(gif_name)['gif_name']
    contours = json.loads(contour)['contour']
    joints = json.loads(joint)['joint']

    target_dir = "web_test/"
    motion_cfg_fn = f'config/motion/{gif_name}.yaml'
    if gif_name == 'hi' or gif_name == 'hurray' or gif_name =='jelly' or gif_name =='lala':
        retarget_file = 'cmu1_pfp_copy'
    elif gif_name == 'jesse_dance' or gif_name =='jazz' or gif_name =='sound':
        retarget_file = 'mixamo_fff'
    elif gif_name == 'jumping_jacks':
        retarget_file = 'cmu1_pfp'
    elif gif_name == 'sun' or gif_name == 'waltz_f':
        retarget_file = 'git'
    else:
        retarget_file = 'fair1_ppf'

    retarget_cfg_fn = f'config/retarget/{retarget_file}.yaml'

    img = await file.read()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)[:, :, ::-1]  # RGB

    #save texture
    print("[INFO] Save texture")
    shape = await save_texture(img, target_dir)

    #save mask
    print("[INFO] Contour to mask")
    contour_to_mask(contours, shape)

    #save joint
    print("[INFO] Save joint")
    save_joint(joints, shape, target_dir)

    annotations_to_animation(target_dir, motion_cfg_fn, retarget_cfg_fn)

    return 'done'

@app.get("/get_gif_list")
def get_gif_list():
    gif_dir = "web_contents/exhibit/"
    gif_files = [f for f in os.listdir(gif_dir) if f.endswith(".gif")]
    return {"gifs": gif_files}

@app.get("/exhibit")
async def exhibit_page(request: Request):
    return templates.TemplateResponse("exhibit.html", {"request": request})

if __name__ == '__main__':
    uvicorn.run( app="main:app", host="0.0.0.0", port=8888, reload=True)