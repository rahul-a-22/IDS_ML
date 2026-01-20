import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/CICIDS2017_sample.csv")

print(df["Label"].value_counts())

df = df[df["Label"] == "BENIGN"]

if df.empty:
    raise ValueError("No BENIGN samples found in dataset.")

X = df.drop("Label", axis=1)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

anomaly_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

anomaly_model.fit(X_scaled)

with open("models/anomaly_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": anomaly_model,
            "scaler": scaler
        },
        f
    )

print("✅ Anomaly model trained successfully using BENIGN traffic")
