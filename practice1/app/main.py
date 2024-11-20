from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.rgb_yuv import RGBtoYUV, YUVtoRGB
from app.services.resize_image import resize_image
from app.services.serpentine import serpentine
from app.services.dct import DCT
from app.services.dwt import DWT
import numpy as np
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


# ---- Pydantic Models for Validation ----
class MatrixData(BaseModel):
    data: list[list[int]]


class DCTData(BaseModel):
    dct_data: list[list[float]]


class IDWTData(BaseModel):
    approx: list[list[float]]
    horiz: list[list[float]]
    vert: list[list[float]]
    diag: list[list[float]]


# ---- RGB to YUV Conversion ----
@app.post("/convert/rgb-to-yuv/")
def convert_rgb_to_yuv(r: int, g: int, b: int):
    converter = RGBtoYUV(r, g, b)
    return {"yuv": converter.conversor()}


# ---- YUV to RGB Conversion ----
@app.post("/convert/yuv-to-rgb/")
def convert_yuv_to_rgb(y: float, u: float, v: float):
    converter = YUVtoRGB(y, u, v)
    return {"rgb": converter.conversor()}


# ---- Resize Image ----
@app.post("/image/resize/")
def resize_image_api(file: UploadFile = File(...), width: int = 320, height: int = 240):
    try:
        input_path = f"/tmp/{file.filename}"
        output_path = f"/tmp/resized_{file.filename}"
        with open(input_path, "wb") as buffer:
            buffer.write(file.file.read())
        resize_image(input_path, output_path, width, height)
        return {"output_image": output_path}
    except Exception as e:
        logger.error(f"Image resizing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Serpentine ----
@app.post("/image/serpentine/")
def serpentine_api(file: UploadFile = File(...)):
    try:
        input_path = f"/tmp/{file.filename}"
        with open(input_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        result = serpentine(input_path)
        return {"serpentine": result}
    except Exception as e:
        logger.error(f"Serpentine processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Discrete Cosine Transform (DCT) ----
@app.post("/dct/")
def apply_dct(matrix: MatrixData):
    try:
        matrix_data = np.array(matrix.data)
        dct_processor = DCT(matrix_data)
        transformed = dct_processor.dct()
        return {"dct": transformed.tolist()}
    except Exception as e:
        logger.error(f"DCT processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Inverse Discrete Cosine Transform (IDCT) ----
@app.post("/idct/")
def apply_idct(dct_data: DCTData):
    try:
        matrix = np.array(dct_data.dct_data)
        dct_processor = DCT(matrix)
        dct_processor.transformed_data = matrix  # Explicitly set transformed data
        reconstructed = dct_processor.idct()
        return {"reconstructed": reconstructed.tolist()}
    except Exception as e:
        logger.error(f"IDCT processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Discrete Wavelet Transform (DWT) ----
@app.post("/dwt/")
def apply_dwt(matrix: MatrixData):
    try:
        matrix_data = np.array(matrix.data)
        dwt_processor = DWT(matrix_data)
        coeffs = dwt_processor.dwt()
        return {
            "approximation": coeffs[0].tolist(),
            "horizontal_detail": coeffs[1].tolist(),
            "vertical_detail": coeffs[2].tolist(),
            "diagonal_detail": coeffs[3].tolist(),
        }
    except Exception as e:
        logger.error(f"DWT processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Inverse Discrete Wavelet Transform (IDWT) ----
@app.post("/idwt/")
def apply_idwt(idwt_data: IDWTData):
    try:
        coeffs = (
            np.array(idwt_data.approx),
            np.array(idwt_data.horiz),
            np.array(idwt_data.vert),
            np.array(idwt_data.diag),
        )
        dwt_processor = DWT(np.zeros((len(idwt_data.approx) * 2, len(idwt_data.approx[0]) * 2)))
        dwt_processor.coeffs = coeffs
        reconstructed = dwt_processor.idwt()
        return {"reconstructed": reconstructed.tolist()}
    except Exception as e:
        logger.error(f"IDWT processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
