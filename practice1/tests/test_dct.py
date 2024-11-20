from fastapi.testclient import TestClient
from app.main import app
import numpy as np

client = TestClient(app)

def test_dct():
    data = [
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 66, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ]
    response = client.post("/dct/", json={"data": data})
    assert response.status_code == 200
    assert "dct" in response.json()

def test_idct():
    dct_data = [
        [231, -48, -36, -32, -36, -24, -26, -20],
        [34, -42, -30, -25, -36, -26, -20, -18],
        [-28, -20, -26, -32, -40, -36, -34, -26],
        [-36, -34, -20, -30, -34, -26, -24, -20],
        [-32, -28, -24, -20, -18, -14, -10, -8],
        [-24, -22, -16, -14, -10, -6, -4, -2],
        [-18, -16, -12, -10, -8, -6, -4, -2],
        [-10, -8, -6, -4, -2, -1, 0, 0],
    ]
    response = client.post("/idct/", json={"dct_data": dct_data})
    assert response.status_code == 200
    assert "reconstructed" in response.json()

