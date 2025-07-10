import base64
import os
import shutil
import uuid
from typing import Literal

import uvicorn
from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse

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


@app.post("/analyze/{leg}")
async def analyze_video(leg: Literal["left", "right"], file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.mov')):
        return JSONResponse(status_code=400, content={"error": "Invalid file format"})

    output_img = f"output_{uuid.uuid4().hex}.jpg"
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    debug_video_path = f"debug_{uuid.uuid4().hex}.mp4"

    file.file.seek(0)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("Wrote temp file:", temp_path)
    print("File size:", os.path.getsize(temp_path), "bytes")

    try:
        result = process_video(temp_path, leg=leg, draw_debug=True, save_debug_path=debug_video_path,
                               output_image_path=output_img)
        if result["max_knee_angle"] is None:
            return JSONResponse(status_code=422, content={"error": "No knee detected."})
        with open(output_img, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        with open(debug_video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        return {
            "max_knee_angle": result["max_knee_angle"],
            "frames_analyzed": result.get("frames_analyzed", None),
            "image_base64": img_base64,
            "video_base64": video_base64
        }

    except Exception as e:
        print("Exception during analysis:", e)
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)

    finally:
        os.remove(temp_path)
        if os.path.exists(output_img):
            os.remove(output_img)
        if os.path.exists(debug_video_path):
            os.remove(debug_video_path)


@app.get("/test")
async def test_get():
    print("Received Request")
    return {"message": "Status: Backend is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
