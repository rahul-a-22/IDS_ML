import pickle
import numpy as np

with open("models/cl_kmeans_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

kmeans = artifacts["model"]
scaler = artifacts["scaler"]

def detect_cl_kmeans(features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    cluster = kmeans.predict(X_scaled)[0]
    distances = kmeans.transform(X_scaled)[0]
    distance = distances[cluster]

    threshold = np.percentile(
        kmeans.transform(kmeans.cluster_centers_), 90
    )

    is_anomaly = distance > threshold

    return {
        "is_anomaly": is_anomaly,
        "cluster": int(cluster),
        "distance": float(distance),
        "threshold": float(threshold),
        "all_distances": distances.tolist()
    }
