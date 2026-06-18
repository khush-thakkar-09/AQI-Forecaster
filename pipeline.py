import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

LAT, LON = 22.5626, 88.3630

# ── Feature order (19 cols) ──
FEATURE_COLS = [
    'pm25', 'pm10', 'no2', 'co', 'so2', 'o3',
    'temp', 'rh', 'wind', 'rain',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'heat_index',
    'festival_flag', 'rush_hour_flag', 'season_flag', 'crop_burning_flag'
]

COLS_TO_SCALE = [
    'pm25', 'pm10', 'no2', 'co', 'so2', 'o3', 
    'temp', 'rh', 'wind', 'rain', 'heat_index'
]

def calculate_heat_index(T, RH):
    HI = (-8.78
          + 1.61 * T
          + 2.34 * RH
          - 0.146 * T * RH
          - 0.012 * T**2
          - 0.016 * RH**2
          + 0.00022 * T**2 * RH
          + 0.00086 * T * RH**2
          - 0.000002 * T**2 * RH**2)
    return HI

def get_season(month):
    if month in [12, 1, 2]:
        return 0   # Winter
    elif month in [3, 4, 5]:
        return 1   # Summer
    elif month in [6, 7, 8, 9]:
        return 2   # Monsoon
    else:
        return 3   # Post-monsoon (Oct, Nov)

def run_pipeline(scaler):
    """
    Fetches real-time weather and air quality data, performs all feature
    engineering steps, scales the numeric features, and returns:
    1. A list of lists of shape (23, 19) containing the scaled features.
    2. A dictionary of the latest raw (unscaled) readings for UI display.
    """
    now = datetime.now()
    start = now - timedelta(hours=23)
    start_date = start.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    # 1. Fetch Air Quality
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5,pm10,ozone,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Asia%2FKolkata"
    )
    
    # 2. Fetch Weather
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,rain,wind_speed_10m"
        f"&wind_speed_unit=ms"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Asia%2FKolkata"
    )

    df_aq = pd.DataFrame(requests.get(aq_url).json()['hourly'])
    df_weather = pd.DataFrame(requests.get(weather_url).json()['hourly'])

    # 3. Merge
    df = pd.merge(df_aq, df_weather, on='time')

    # 4. Rename Columns
    df.rename(columns={
        'pm2_5'              : 'pm25',
        'pm10'               : 'pm10',
        'ozone'              : 'o3',
        'carbon_monoxide'    : 'co',
        'nitrogen_dioxide'   : 'no2',
        'sulphur_dioxide'    : 'so2',
        'temperature_2m'     : 'temp',
        'relative_humidity_2m': 'rh',
        'rain'               : 'rain',
        'wind_speed_10m'     : 'wind'
    }, inplace=True)

    # 5. Filter to last 23 hours
    current_hour_str = now.strftime("%Y-%m-%dT%H:00")
    df['time'] = pd.to_datetime(df['time'])
    cutoff = pd.to_datetime(current_hour_str)
    df = df[df['time'] <= cutoff].tail(23).reset_index(drop=True)

    if len(df) < 23:
        raise ValueError(f"Insufficient data returned: expected 23 rows, got {len(df)}")

    # 6. Feature Engineering
    df['co'] = df['co'] / 1000  # Convert µg/m³ to mg/m³
    
    # Date variables
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"]   = df["time"].dt.day
    df["hour"]  = df["time"].dt.hour

    # Sin/Cos transformations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Heat Index
    df["heat_index"] = calculate_heat_index(df["temp"], df["rh"])

    # Flags
    festival_dates = pd.to_datetime(["2026-11-08", "2026-10-20"]).date
    df["festival_flag"] = df["time"].dt.date.isin(festival_dates).astype(int)

    rush_hours = list(range(7, 11)) + list(range(17, 21))
    df["rush_hour_flag"] = df["hour"].isin(rush_hours).astype(int)

    df["season_flag"] = df["month"].apply(get_season)

    df["crop_burning_flag"] = (
        (df["month"].isin([10, 11])) & 
        ~((df["month"] == 10) & (df["day"] < 15))
    ).astype(int)

    # Capture the latest raw values before transformations & scaling
    latest_row = df.iloc[-1]
    raw_data = {
        "time": latest_row["time"].strftime("%Y-%m-%d %H:%M"),
        "pm25": float(latest_row["pm25"]),
        "pm10": float(latest_row["pm10"]),
        "no2": float(latest_row["no2"]),
        "co": float(latest_row["co"]),
        "so2": float(latest_row["so2"]),
        "o3": float(latest_row["o3"]),
        "temp": float(latest_row["temp"]),
        "rh": float(latest_row["rh"]),
        "wind": float(latest_row["wind"]),
        "rain": float(latest_row["rain"]),
        "heat_index": float(latest_row["heat_index"]),
        "festival_flag": int(latest_row["festival_flag"]),
        "rush_hour_flag": int(latest_row["rush_hour_flag"]),
        "season_flag": int(latest_row["season_flag"]),
        "crop_burning_flag": int(latest_row["crop_burning_flag"])
    }

    # Apply Log transformations
    df["pm25"] = np.log1p(df["pm25"])
    df["pm10"] = np.log1p(df["pm10"])

    # Scale numeric columns
    for i, col in enumerate(COLS_TO_SCALE):
        df[col] = (df[col] - scaler.mean_[i]) / scaler.scale_[i]

    # Order columns exactly as model expects
    df_features = df[FEATURE_COLS]

    return df_features.values.tolist(), raw_data
