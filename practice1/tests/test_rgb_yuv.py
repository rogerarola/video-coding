import pytest
from app.services.rgb_yuv import RGBtoYUV, YUVtoRGB
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rgb_to_yuv():
    response = client.post("/convert/rgb-to-yuv/?r=16&g=128&b=128")
    assert response.status_code == 200
    assert response.json() == {"yuv": pytest.approx([97.625, 145.125, 79.5], rel=1e-1)}

def test_yuv_to_rgb():
    response = client.post("/convert/yuv-to-rgb/?y=97.625&u=145.125&v=79.5")
    assert response.status_code == 200
    assert response.json() == {"rgb": pytest.approx([17.6055, 127.729, 129.5526], rel=1e-1)}
