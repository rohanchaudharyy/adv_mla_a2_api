from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import datetime
from joblib import load
import numpy as np

app = FastAPI(
    title="Sydney Weather Forecast API",
    description="API to predict rain in 7 days and 3-day cumulative precipitation in Sydney."
)

# Load models (replace with your actual model filenames)
rain_model = load("models/model_7.pkl")
precip_model = load("models/model_3.pkl")

rain_features = [
    'temperature_2m_', 'relative_humidity_2m_', 'dew_point_2m_',
    'apparent_temperature_', 'cloud_cover_', 'surface_pressure_hPa',
    'wind_speed_100m_km/h', 'wind_gusts_10m_km/h',
    'vapour_pressure_deficit_kPa', 'hour', 'day', 'month', 'weekday',
    'temperature_2m__lag1', 'temperature_2m__lag3',
    'relative_humidity_2m__lag1', 'relative_humidity_2m__lag3',
    'temperature_2m__roll3', 'relative_humidity_2m__roll3'
]

precip_features = rain_features  # same features for simplicity

def create_features(input_date: datetime.datetime, feature_list):
    df = pd.DataFrame({col: [0] for col in feature_list})
    # Fill time-based features
    df["hour"] = input_date.hour
    df["day"] = input_date.day
    df["month"] = input_date.month
    df["weekday"] = input_date.weekday()
    return df

@app.get("/")
def read_root():
    return {
        "project": "Sydney Weather Forecast",
        "description": "Predicting rain in 7 days and 3-day cumulative precipitation.",
        "endpoints": {
            "health": "/health/",
            "predict_rain": "/predict/rain/?date=YYYY-MM-DD",
            "predict_precipitation": "/predict/precipitation/fall/?date=YYYY-MM-DD"
        },
        "github_repo": "https://github.com/rohanchaudharyy/adv_mla_a2_api"
    }

@app.get("/health/")
def health_check():
    return {"status": "API is running successfully!"}

@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., description="Input date in YYYY-MM-DD format")):
    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    features = create_features(input_date, rain_features)
    try:
        will_rain = bool(rain_model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "input_date": date,
        "prediction": {
            "date": (input_date + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
            "will_rain": will_rain
        }
    }

@app.get("/predict/precipitation/fall/")
def predict_precipitation(date: str = Query(..., description="Input date in YYYY-MM-DD format")):
    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    features = create_features(input_date, precip_features)
    try:
        precipitation_fall = float(precip_model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "input_date": date,
        "prediction": {
            "start_date": (input_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": (input_date + datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
            "precipitation_fall": precipitation_fall
        }
    }

# To run: uvicorn app.main:app --reload
