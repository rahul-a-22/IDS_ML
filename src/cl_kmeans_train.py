import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 11

df = pd.read_csv("data/CICIDS2017_sample.csv")

print(df["Label"].value_counts())

df = df[df["Label"] == "BENIGN"]

X = df.drop("Label", axis=1)

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

with open("models/cl_kmeans_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": kmeans,
            "scaler": scaler,
            "n_clusters": N_CLUSTERS
        },
        f
    )

print("✅ CL-KMeans trained on BENIGN traffic")
