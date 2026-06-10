import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib

from src.data.preprocess import clean_raw_data, apply_outlier_caps
from src.features.build_features import build_features

app = FastAPI(title="Churn Prediction API")

# Load models on startup
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
preprocessor = None
model = None
cap_values = None

try:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "Voting.joblib")
    print("[SUCCESS] Models loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load models. Make sure you run the pipeline first. Error: {e}")

try:
    cap_path = MODELS_DIR / "cap_values.joblib"
    if cap_path.exists():
        cap_values = joblib.load(cap_path)
        print("[SUCCESS] Outlier cap values loaded successfully!")
    else:
        print("[WARNING] cap_values.joblib not found. Outlier capping will be skipped.")
except Exception as e:
    print(f"[WARNING] Failed to load outlier cap values: {e}")

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # 1. Convert incoming JSON to DataFrame
        df_raw = pd.DataFrame([data.dict()])
        
        # 2. Apply cleaning (handles types and missing values)
        df_cleaned = clean_raw_data(df_raw)
        
        # 3. Apply feature engineering
        df_features = build_features(df_cleaned)
        
        # 3.5. Apply outlier capping if available
        if cap_values is not None:
            df_features = apply_outlier_caps(df_features, cap_values)
        
        # 4. Preprocess (StandardScaler + OneHotEncoder)
        X_processed = preprocessor.transform(df_features)
        
        # 5. Predict using Voting Classifier
        churn_prob = model.predict_proba(X_processed)[0][1]
        is_churn = int(model.predict(X_processed)[0])
        
        return {
            "status": "success",
            "churn_probability": float(churn_prob),
            "churn_prediction": is_churn,
            "risk_level": "High" if churn_prob > 0.6 else ("Medium" if churn_prob > 0.4 else "Low")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
