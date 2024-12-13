from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.video_converter import video_converter
from app.services.encoding_ladder import encoding_ladder
import os
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Dockerized FastAPI!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/video/convert/")
async def convert_video(file: UploadFile = File(...), format: str = "vp8"):
    try:
        # Save the uploaded file to a temporary location
        input_path = f"/tmp/{file.filename}"
        with open(input_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Ensure the format is valid
        valid_formats = ["vp8", "vp9", "h265", "av1"]
        if format.lower() not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

        # Perform the video conversion
        output_message = video_converter(input_path, format)

        # Clean up the input file
        os.remove(input_path)

        return {"message": output_message}
    except Exception as e:
        logger.error(f"Video conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video/encoding-ladder/")
async def generate_encoding_ladder(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        input_path = f"/tmp/{file.filename}"
        with open(input_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Generate the encoding ladder
        output_files = encoding_ladder(input_path)

        # Clean up the input file
        os.remove(input_path)

        # Return the generated file paths
        return {"message": "Encoding ladder generated successfully", "files": output_files}
    except Exception as e:
        logger.error(f"Encoding ladder generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))