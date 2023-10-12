from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from image_to_annotations import image_to_annotations
from fastapi import FastAPI, UploadFile, File, Request
from starlette.responses import HTMLResponse
import shutil
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# uvicorn annotationAPI:app --reload
# http://127.0.0.1:8001/process_image?img_fn=dorosi/image.png&out_dir=dorosi


app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/upload/")
def upload_file_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    with Path("web_test/" + file.filename).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # You can immediately process the image after uploading if you want
    image_paths = image_to_annotations("web_test/" + file.filename, "web_test")
    return templates.TemplateResponse("display.html", {"request": request, "images": image_paths})

    
@app.get("/process_image/")
def process_image(request: Request, img_fn: str, out_dir: str):
    image_paths = image_to_annotations(img_fn, out_dir)
    print(image_paths)
    return templates.TemplateResponse("display.html", {"request": request, "images": image_paths})

@app.get("/get_image/{image_path:path}")
def get_image(image_path: str):
    return FileResponse(image_path)
