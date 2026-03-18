import numpy as np
import pandas as pd
import pickle

# Load feature names from trained model
with open("models/ids_multi_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

feature_names = artifacts["feature_names"]
NUM_FEATURES = len(feature_names)
NUM_SAMPLES = 1000

np.random.seed(42)

data = []

for _ in range(NUM_SAMPLES):
    row = []

    for i in range(NUM_FEATURES):
        if i % 3 == 0:
            value = np.random.uniform(50, 500)       # small irregular
        elif i % 3 == 1:
            value = np.random.uniform(1000, 9000)    # mid anomaly
        else:
            value = np.random.uniform(0, 50)         # sparse noise

        row.append(round(value, 4))

    data.append(row)

df = pd.DataFrame(data, columns=feature_names)

df.to_csv("data/zero_day_samples.csv", index=False)

print("✅ zero_day_samples.csv generated successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
