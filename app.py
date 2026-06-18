from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from tensorflow import keras
from model import ILSTMCell, ILSTMLayer

# ── Feature order (19 cols, matches training CSV minus AQI) ──
FEATURE_COLS = ['pm25',
 'pm10',
 'no2',
 'co',
 'so2',
 'o3',
 'temp',
 'rh',
 'wind',
 'rain',
 'hour_sin',
 'hour_cos',
 'month_sin',
 'month_cos',
 'heat_index',
 'festival_flag',
 'rush_hour_flag',
 'season_flag',
 'crop_burning_flag']

WINDOW       = 23   # timesteps
N_FEATURES   = 19
AQI_SCALER_IDX = 11  # AQI's position in the scaler (scaler was fit on 12 cols: 11 features + AQI)

# ── Load artefacts once at startup ────────────────────────────
scaler = joblib.load('scaler.pkl')
model  = keras.models.load_model(
    'model_3.keras',
    custom_objects={'ILSTMCell': ILSTMCell, 'ILSTMLayer': ILSTMLayer},
)

app = FastAPI(title='AQI Forecaster API')


# ── Schemas ───────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Pre-processed input: a 23x19 matrix (list of 23 lists, each with 19 floats).
    Values must already be scaled / feature-engineered exactly like training.
    """
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predicted_aqi: float

# ── Endpoints ─────────────────────────────────────────────────

@app.post('/predict', response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Accepts a pre-processed 23x19 feature matrix and returns predicted AQI.
    """
    arr = np.array(req.features, dtype=np.float32)
    if arr.shape != (WINDOW, N_FEATURES):
        raise HTTPException(
            status_code=422,
            detail=f'Expected shape ({WINDOW}, {N_FEATURES}), got {arr.shape}',
        )

    X = arr.reshape(1, WINDOW, N_FEATURES)
    y_scaled = model.predict(X, verbose=0).flatten()[0]

    # Inverse-scale: AQI is at index AQI_SCALER_IDX in the scaler
    aqi_mean  = scaler.mean_[AQI_SCALER_IDX]
    aqi_scale = scaler.scale_[AQI_SCALER_IDX]
    aqi  = y_scaled * aqi_scale + aqi_mean

    return PredictionResponse(predicted_aqi=round(aqi, 2))


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/info')
def info():
    """Return expected input shape and feature order."""
    return {
        'window': WINDOW,
        'n_features': N_FEATURES,
        'feature_order': FEATURE_COLS,
        'note': 'All numeric features must be StandardScaler-transformed. '
                'pm25 and pm10 must be log1p-transformed before scaling. '
                'co must be in mg/m³ (divide µg/m³ by 1000) before scaling.',
    }