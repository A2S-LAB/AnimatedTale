from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse
from fastapi import FastAPI, UploadFile, File, Request, status, Form
from fastapi.staticfiles import StaticFiles
import os
import cv2
import uvicorn

from pydantic import BaseModel
import shutil
from pathlib import Path
import logging

from annotations_to_animation import annotations_to_animation
from utils import auto_bbox, predict_mask, predict_joint

logging.basicConfig(level=logging.INFO)
counter = 0

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/css", StaticFiles(directory="templates/css/"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/get_image/{image_path:path}")
def get_image(image_path: str):
    return FileResponse(image_path)

@app.get("/")
def main_page(request: Request, video: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "video": video})

@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "video": None})

@app.post("/process_upload")
async def process_upload(request: Request, file: UploadFile = File(...)):
    target_dir = "web_test/"
    with Path(target_dir + file.filename).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predict_mask(target_dir + file.filename, target_dir)
    predict_joint(target_dir + file.filename, target_dir)

    return templates.TemplateResponse("mask.html", {"request": request})

@app.get("/mask")
async def mask(request: Request):
    return templates.TemplateResponse("mask.html", {"request": request})

@app.get("/joint_overlay")
async def joint_overlay(request: Request):
    return templates.TemplateResponse("joint_overlay.html", {"request": request})

@app.post("/make_gif")
async def make_gif(gif_name: str = Form(...)):
    target_dir = "web_test/"
    motion_cfg_fn = f'config/motion/{gif_name}.yaml'
    if gif_name == 'hi' or gif_name == 'hurray' or gif_name =='jelly':
        retarget_file = 'cmu1_pfp_copy'
    elif gif_name == 'jesse_dance':
        retarget_file = 'mixamo_fff'
    elif gif_name == 'jumping_jacks':
        retarget_file = 'cmu1_pfp'
    else:
        retarget_file = 'fair1_ppf'
    retarget_cfg_fn = f'config/retarget/{retarget_file}.yaml'
    annotations_to_animation(target_dir, motion_cfg_fn, retarget_cfg_fn)
    return RedirectResponse(url="/confirm", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/confirm")
async def confirm(request: Request):
    return templates.TemplateResponse("confirm.html", {"request": request})

@app.post("/move_gif")
async def move_gif():
    global counter

    video_name = f'video_{counter}.gif'
    counter = counter % 10 + 1
    shutil.copy("web_test/" + 'video.gif', "web_contents/exhibit/" + video_name)
    return RedirectResponse(url="/upload?from_confirm=true", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/get_gif_list")
def get_gif_list():
    gif_dir = "web_contents/exhibit/"
    gif_files = [f for f in os.listdir(gif_dir) if f.endswith(".gif")]
    return {"gifs": gif_files}

@app.get("/exhibit")
async def exhibit_page(request: Request):
    return templates.TemplateResponse("exhibit.html", {"request": request})

@app.post("/bbox")
async def bbox():
    image = cv2.imread("web_test/image.png")
    bbox = auto_bbox(image)
    return {"bbox": bbox.tolist()}

@app.get("/motion")
async def motion(request: Request):
    return templates.TemplateResponse("motion.html", {"request": request})

if __name__ == '__main__':
    uvicorn.run( app="api:app", host="0.0.0.0", port=8886, reload=True)