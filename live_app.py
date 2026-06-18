from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import joblib
from tensorflow import keras
from model import ILSTMCell, ILSTMLayer
from pipeline import run_pipeline
import os

# ── Setup and Model Loading ───────────────────────────────────
WINDOW = 23
N_FEATURES = 19
AQI_SCALER_IDX = 11

scaler = joblib.load('scaler.pkl')
model  = keras.models.load_model(
    'model_3.keras',
    custom_objects={'ILSTMCell': ILSTMCell, 'ILSTMLayer': ILSTMLayer}
)

app = FastAPI(title='AQI Live Forecaster API')

# Enable CORS so your Vercel frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────

class PredictionLiveResponse(BaseModel):
    predicted_aqi: float
    raw_data: Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
def serve_dashboard():
    """
    Serves the index.html user interface.
    """
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
        
    with open(index_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get('/predict-live', response_model=PredictionLiveResponse)
def predict_live():
    """
    Fetches raw weather/AQI inputs from Open-Meteo, runs the preprocessing 
    pipeline, feeds the scaled matrix to the model, and returns the 
    prediction along with the latest raw values.
    """
    try:
        scaled_features, raw_data = run_pipeline(scaler)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing data collection pipeline: {str(e)}"
        )

    # Reshape features to (1, 23, 19) for model input
    X = np.array(scaled_features, dtype=np.float32).reshape(1, WINDOW, N_FEATURES)
    y_scaled = model.predict(X, verbose=0).flatten()[0]

    # Inverse-scale the prediction back to actual AQI range
    aqi_mean  = scaler.mean_[AQI_SCALER_IDX]
    aqi_scale = scaler.scale_[AQI_SCALER_IDX]
    aqi  = y_scaled * aqi_scale + aqi_mean

    return PredictionLiveResponse(
        predicted_aqi=round(float(aqi), 2),
        raw_data=raw_data
    )


@app.get('/health')
def health():
    return {'status': 'ok'}
