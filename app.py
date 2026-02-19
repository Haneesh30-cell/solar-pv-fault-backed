from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model and preprocessing objects
model = load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):

    if len(data.features) != 18:
        return {"error": "Exactly 18 features required"}

    features = np.array([data.features])
    features = np.nan_to_num(features)
    features = scaler.transform(features)

    prediction = model.predict(features)
    pred_class = np.argmax(prediction, axis=1)

    label = label_encoder.inverse_transform(pred_class)[0]
    confidence = float(np.max(prediction) * 100)

    return {
        "prediction": label,
        "confidence": round(confidence, 2)
    }
