from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi import Body

from pydantic import BaseModel
import shutil
from pathlib import Path
import logging

from image_to_annotations import image_to_annotations
from image_to_animation import ani_main
from chatGPTAPI import createStory

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
# uvicorn API:app --reload
# lsof -i:8000
# http://127.0.0.1:8001/process_image?img_fn=dorosi/image.png&out_dir=dorosi

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
        with Path("web_test/" + file.filename).open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ani_main("web_test/" + file.filename, "web_test")
        video_path = 'video.gif'
        return templates.TemplateResponse("upload.html", {"request": request, "video": video_path})
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