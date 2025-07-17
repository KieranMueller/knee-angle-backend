import base64
import os
import shutil
import uuid
from typing import Literal

import uvicorn
from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, FileResponse

from core_analyzer import process_video

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # change back to origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TEMP_VIDEO_DIR = "temp_videos"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_video(leg: Literal["left", "right"],
                        orientation: Literal["landscape", "portrait"],
                        file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.mov')):
        return JSONResponse(status_code=400, content={"error": "Invalid file format"})

    output_img = f"output_{uuid.uuid4().hex}.jpg"
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    debug_filename = f"debug_{uuid.uuid4().hex}.mp4"
    debug_video_path = os.path.join(TEMP_VIDEO_DIR, debug_filename)

    file.file.seek(0)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("Wrote temp file:", temp_path)
    print("File size:", os.path.getsize(temp_path), "bytes")

    try:
        # debug video does not output if save_debug_path
        result = process_video(temp_path, leg=leg, draw_debug=True, save_debug_path=debug_video_path,
                               output_image_path=output_img, orientation=orientation)
        if result["max_knee_angle"] is None:
            return JSONResponse(status_code=422, content={"error": "No knee detected."})
        with open(output_img, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        return {
            "max_knee_angle": result["max_knee_angle"],
            "frames_analyzed": result.get("frames_analyzed", None),
            "average_largest_knee_angles": result["average_largest_knee_angles"],
            "image_base64": img_base64,
            "video_filename": debug_filename
        }

    except Exception as e:
        print("Exception during analysis:", e)
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)

    finally:
        os.remove(temp_path)
        if os.path.exists(output_img):
            os.remove(output_img)


@app.get("/debug-video/{path}")
async def get_debug_video(path: str):
    print(f"Received request to fetch debug video with path {path}")
    video_path = os.path.join(TEMP_VIDEO_DIR, path)
    if not os.path.exists(video_path):
        return {'error': 'File not found'}
    # If temp vids start to pile up and
    return FileResponse(video_path, media_type="video/mp4", filename=path)


@app.get("/test")
async def test_get():
    print("Received Request")
    return {"message": "Status: Backend is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)