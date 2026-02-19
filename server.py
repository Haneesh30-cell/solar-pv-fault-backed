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

    features = np.array(data.features).reshape(1, -1)

    if features.shape[1] != 18:
        return {"error": "Model expects exactly 18 features"}

    features = scaler.transform(features)

    prediction = model.predict(features)
    pred_class = np.argmax(prediction, axis=1)

    fault = label_encoder.inverse_transform(pred_class)[0]
    confidence = float(np.max(prediction) * 100)

    return {
        "prediction": fault,
        "confidence": round(confidence, 2)
    }
