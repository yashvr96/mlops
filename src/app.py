from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os
import logging
import time
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define input data model
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 0,
                    "ca": 0,
                    "thal": 1
                }
            ]
        }
    }

app = FastAPI(title="Heart Disease Prediction API")

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# Load model and scaler
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} Method: {request.method} Status: {response.status_code} Duration: {process_time:.4f}s")
    return response

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running."}

@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    if not model or not scaler:
        logger.error("Model or Scaler not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Log input (be careful with PII in production, but okay for this dataset)
        logger.info(f"Received prediction request: {input_data}")
        
        # Convert input to DataFrame
        data = input_data.dict()
        df = pd.DataFrame([data])
        
        # Scale
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[:, 1] if hasattr(model, "predict_proba") else [0.0]
        
        result = {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "risk": "High" if prediction[0] == 1 else "Low"
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
