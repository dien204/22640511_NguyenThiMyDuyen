import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


bundle = joblib.load("models/model.pkl")
model = bundle["model"]
feature_info = bundle["feature_info"]

app = FastAPI(title="MLOps Prediction API")


class InputData(BaseModel):
    data: dict


@app.get("/")
def home():
    return {
        "message": "API is running",
        "features": list(feature_info.keys())
    }


@app.post("/predict")
def predict(input_data: InputData):
    X = pd.DataFrame([input_data.data])
    prediction = model.predict(X)[0]
    return {"prediction": str(prediction)}
