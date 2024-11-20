from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_serpentine():
    input_image_path = "/Users/rogerarolaplanas/Documents/GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/image.jpg"
    with open(input_image_path, "rb") as file:
        response = client.post(
            "/image/serpentine/",
            files={"file": ("image.jpg", file, "image/jpeg")},
        )
    assert response.status_code == 200
    assert "serpentine" in response.json()
