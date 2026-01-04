from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

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

app = FastAPI(title="Heart Disease Prediction API")

# Load model and scaler
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running."}

@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Convert input to DataFrame
        data = input_data.dict()
        df = pd.DataFrame([data])
        
        # Ensure column order matches training (simplified here, assumes correct order)
        # Ideally we'd use the feature names from the model/scaler if stored.
        # For now, we rely on Pydantic's order if it matches the CSV columns (minus target).
        # We need to be careful with column alignment.
        # Let's assume the input fields match the feature order.
        
        # Scale
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[:, 1] if hasattr(model, "predict_proba") else [0.0]
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "risk": "High" if prediction[0] == 1 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
