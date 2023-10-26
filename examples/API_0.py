from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse
from fastapi import FastAPI, UploadFile, File, Request, WebSocket, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
import os

from pydantic import BaseModel
import shutil
from pathlib import Path
import logging

from image_to_annotations1 import image_to_annotations
from annotations_to_animation import annotations_to_animation
from image_to_animation import ani_main
from chatGPTAPI import createStory
from diffusion import makeBackground

# uvicorn API_0:app --reload
# lsof -i:8000

logging.basicConfig(level=logging.INFO)
counter = 0  # Global counter for gif naming

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

templates = Jinja2Templates(directory="templates")

@app.get("/get_image/{image_path:path}")
def get_image(image_path: str):
    return FileResponse(image_path)

@app.get("/get_video/{video_path:path}")
def get_video(video_path: str):
    return FileResponse(video_path)

@app.get("/")
def main_page(request: Request, video: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "video": video})

@app.get("/story")
async def story_page(request: Request):
    return templates.TemplateResponse("story.html", {"request": request})

class StoryRequest(BaseModel):
    prompt: str

@app.post("/create_story")
async def create_story_endpoint(request: StoryRequest):
    try:
        story = createStory(request.prompt)
        return {"story": story}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Story creation failed.")

@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "video": None})

@app.post("/process_upload")
async def process_upload(request: Request, file: UploadFile = File(...)):
    target_dir = "web_test/"
    with Path(target_dir + file.filename).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    image_to_annotations(target_dir + file.filename, target_dir)
    
    return templates.TemplateResponse("texture.html", {"request": request})

@app.get("/texture")
async def texture(request: Request):
    return templates.TemplateResponse("texture.html", {"request": request})

@app.get("/mask")
async def mask(request: Request):
    return templates.TemplateResponse("mask.html", {"request": request})

@app.get("/joint_overlay")
async def joint_overlay(request: Request):
    return templates.TemplateResponse("joint_overlay.html", {"request": request})

@app.post("/make_gif")
async def make_gif():
    target_dir = "web_test/"
    motion_cfg_fn = 'config/motion/jumping_jacks.yaml'
    retarget_cfg_fn = 'config/retarget/cmu1_pfp.yaml'
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