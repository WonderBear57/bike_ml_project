import math

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Bike Demand Forecasting API",
    description="Predicts hourly bike-sharing demand and recommends adding or removing bikes",
    version="1.0"
)

# lets the JavaScript page (hosted somewhere else) call this API from the browser
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Loaded ONCE, when the server starts (not on every request)
model           = joblib.load("demand_model.pkl")      # Random Forest
scaler          = joblib.load("scaler.pkl")            # StandardScaler fitted on the training data
feature_columns = joblib.load("feature_columns.pkl")   # the exact 47 columns, in the exact order
slots           = joblib.load("slots.pkl")             # usual demand and demand profile of each (weekday, hour)

SAFETY_FACTOR = 1.57     # same buffer as in the notebook (bikes_needed = predicted demand x 1.57)


class DemandInput(BaseModel):
    season: int     = Field(ge=1, le=4, description="1 = winter, 2 = spring, 3 = summer, 4 = fall")
    yr: int         = Field(default=1, ge=0, le=1, description="0 = 2011, 1 = 2012")
    mnth: int       = Field(ge=1, le=12, description="month, 1 to 12")
    hr: int         = Field(ge=0, le=23, description="hour of the day, 0 to 23")
    holiday: int    = Field(ge=0, le=1, description="1 = holiday, 0 = not")
    weekday: int    = Field(ge=0, le=6, description="0 = Sunday, 1 = Monday ... 6 = Saturday")
    workingday: int = Field(ge=0, le=1, description="1 = neither weekend nor holiday, 0 = otherwise")
    weathersit: int = Field(ge=1, le=4, description="1 = clear, 2 = mist/cloudy, 3 = light rain/snow, 4 = heavy rain/snow")
    temp_celsius: float     = Field(ge=-8, le=39, description="temperature in degrees Celsius")
    humidity_percent: float = Field(ge=1, le=100, description="humidity in percent")
    windspeed: float        = Field(ge=0, le=67, description="wind speed (0 to 67)")


# same rule as in the notebook
def get_recommendation(demand, usual):
    difference = demand - usual
    change_percent = difference / usual * 100

    if change_percent > 15 and difference >= 10:
        return "Add bikes"
    elif change_percent < -15 and difference <= -10:
        return "Remove bikes"
    else:
        return "No change"


@app.get("/")
def home():
    return {
        "message": "Bike Demand Forecasting API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict_demand(data: DemandInput):

    # 1) one row with the raw values; temp / hum / windspeed go back to the 0-1 scale used in the dataset
    row = pd.DataFrame([{
        "season": data.season,
        "yr": data.yr,
        "mnth": data.mnth,
        "hr": data.hr,
        "holiday": data.holiday,
        "weekday": data.weekday,
        "workingday": data.workingday,
        "weathersit": data.weathersit,
        "temp": (data.temp_celsius + 8) / 47,
        "hum": data.humidity_percent / 100,
        "windspeed": data.windspeed / 67,
    }])

    # 2) one-hot encode (no drop_first here: reindex below removes the first category of each column)
    row = pd.get_dummies(row, columns=["season", "weathersit", "hr", "mnth"])

    # 3) same 47 columns in the same order as in training (missing columns are filled with 0)
    row = row.reindex(columns=feature_columns, fill_value=0)

    # 4) scale and predict
    predicted = max(0, round(float(model.predict(scaler.transform(row))[0])))     # demand cannot be negative

    # usual demand and demand profile of this weekday and hour
    slot    = slots[(slots["weekday"] == data.weekday) & (slots["hr"] == data.hr)].iloc[0]
    usual   = max(1, round(float(slot["avg_demand"])))
    profile = slot["demand_profile"]

    return {
        "predicted_demand": predicted,
        "usual_demand": usual,
        "demand_profile": profile,
        "change_percent": round((predicted - usual) / usual * 100, 1),
        "recommendation": get_recommendation(predicted, usual),
        "bikes_needed": math.ceil(predicted * SAFETY_FACTOR)
    }
