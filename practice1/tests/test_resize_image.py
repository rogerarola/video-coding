import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_resize_image():
    input_image_path = "/Users/rogerarolaplanas/Documents/GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/image.jpg"
    with open(input_image_path, "rb") as file:
        response = client.post(
            "/image/resize/",
            files={"file": ("image.jpg", file, "image/jpeg")},
            data={"width": 320, "height": 240},
        )
    assert response.status_code == 200
    assert "/tmp/resized_image.jpg" in response.json()["output_image"]
