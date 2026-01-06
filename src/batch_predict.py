import pandas as pd
import numpy as np
import pickle

MODEL_PATH = "models/ids_multi_model.pkl"
INPUT_CSV = "data/test.csv"
OUTPUT_CSV = "data/test_output.csv"

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

models = artifacts["models"]
accuracies = artifacts["accuracies"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

df = pd.read_csv(INPUT_CSV)

df = df.reindex(columns=feature_names, fill_value=0)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X_scaled = scaler.transform(df.values)

predictions = {}
probabilities = {}

for name, model in models.items():
    pred = model.predict(X_scaled)
    predictions[name] = label_encoder.inverse_transform(pred)

    if hasattr(model, "predict_proba"):
        probabilities[name] = model.predict_proba(X_scaled)

pred_df = pd.DataFrame(predictions)

final_labels = []

for i in range(len(df)):
    votes = {}
    for model_name, label in pred_df.iloc[i].items():
        votes[label] = votes.get(label, 0) + accuracies[model_name]
    final_labels.append(max(votes, key=votes.get))

df["IDS_Prediction"] = final_labels

df.to_csv(OUTPUT_CSV, index=False)

print("✅ Batch prediction completed")
print(f"📄 Output saved to {OUTPUT_CSV}")
