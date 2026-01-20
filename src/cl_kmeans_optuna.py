import pandas as pd
import numpy as np
import optuna
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/CICIDS2017_sample.csv")
df = df[df["Label"] == "BENIGN"]

X = df.drop("Label", axis=1)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def objective(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 20)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("✅ Best number of clusters:", study.best_params["n_clusters"])
print("✅ Best silhouette score:", study.best_value)
