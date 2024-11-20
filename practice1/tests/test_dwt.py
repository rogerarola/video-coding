import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_dwt():
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
    response = client.post("/dwt/", json={"data": data})
    assert response.status_code == 200
    assert "approximation" in response.json()

def test_idwt():
    coeffs = {
        "approx": np.random.rand(4, 4).tolist(),
        "horiz": np.random.rand(4, 4).tolist(),
        "vert": np.random.rand(4, 4).tolist(),
        "diag": np.random.rand(4, 4).tolist()
    }
    response = client.post("/idwt/", json=coeffs)
    assert response.status_code == 200
    assert "reconstructed" in response.json()
