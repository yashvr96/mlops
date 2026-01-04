from fastapi.testclient import TestClient
from src.app import app
import pytest

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Heart Disease Prediction API is running."}

def test_predict_valid():
    # Example valid input
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, 
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, 
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "risk" in response.json()

def test_predict_invalid():
    # Missing fields
    payload = {"age": 63}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
