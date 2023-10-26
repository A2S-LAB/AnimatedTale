from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

from pydantic import BaseModel
import shutil
from pathlib import Path
import logging

from image_to_annotations1 import image_to_annotations
from annotations_to_animation import annotations_to_animation
from image_to_animation import ani_main
from chatGPTAPI import createStory
from diffusion import makeBackground


# uvicorn API:app --reload
# lsof -i:8000
# http://127.0.0.1:8001/process_image?img_fn=dorosi/image.png&out_dir=dorosi

logging.basicConfig(level=logging.INFO)
counter = 0  # Global counter for gif naming
connected_websockets = []  # Global list to keep track of connected WebSockets

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def main_page(request: Request, video: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "video": video})

@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "video": None})

@app.post("/process_upload")
async def process_upload(request: Request, file: UploadFile = File(...)):
    try:
        target_dir = "web_test/"
        with Path(target_dir + file.filename).open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_to_annotations(target_dir + file.filename, target_dir)
        motion_cfg_fn = 'config/motion/zombie.yaml'
        retarget_cfg_fn = 'config/retarget/fair1_ppf.yaml'
        annotations_to_animation(target_dir, motion_cfg_fn, retarget_cfg_fn)
        
        global counter
        counter += 1
        video_name = f'video_{counter}.gif'
        shutil.copy("web_test/" + 'video.gif', "web_contents/exhibit/" + video_name)

        # Notify all connected clients via WebSocket about the new gif
        for ws in connected_websockets:
            await ws.send_text("new_gif")

        return templates.TemplateResponse("upload.html", {"request": request, "video": 'video.gif'})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

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

@app.get("/get_image/{image_path:path}")
def get_image(image_path: str):
    return FileResponse(image_path)

@app.get("/get_video/{video_path:path}")
def get_video(video_path: str):
    return FileResponse(video_path)

import os

@app.get("/get_gif_list")
def get_gif_list():
    gif_dir = "web_contents/exhibit/"
    gif_files = [f for f in os.listdir(gif_dir) if f.endswith(".gif")]
    return {"gifs": gif_files}

@app.get("/exhibit")
async def exhibit_page(request: Request):
    return templates.TemplateResponse("exhibit.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        connected_websockets.remove(websocket)
