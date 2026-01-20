import pickle
import numpy as np

with open("models/anomaly_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

anomaly_model = artifacts["model"]
scaler = artifacts["scaler"]

def detect_anomaly(features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    score = anomaly_model.decision_function(X_scaled)[0]
    pred = anomaly_model.predict(X_scaled)[0]

    is_anomaly = True if pred == -1 else False

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": float(score)
    }
print("✅ Anomaly detection model loaded from models/anomaly_model.pkl")