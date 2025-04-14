import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "risk_prediction_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return {"model": model, "scaler": scaler}

def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)["features"]
        return np.array(data).reshape(1, -1)
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model_bundle):
    scaled = model_bundle["scaler"].transform(input_data)
    prediction = model_bundle["model"].predict(scaled)
    return prediction.tolist()
