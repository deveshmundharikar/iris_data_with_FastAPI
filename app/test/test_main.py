from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# app/test/test_main.py
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Classifier API"}


def test_predict():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json()
