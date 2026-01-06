import pickle
import numpy as np
import os
from collections import Counter

MODEL_PATH = os.path.join("models", "ids_multi_model.pkl")

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

models = artifacts["models"]
accuracies = artifacts["accuracies"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

EXPECTED = len(feature_names)

def normalize_features(features):
    features = list(map(float, features))
    if len(features) < EXPECTED:
        features.extend([0.0] * (EXPECTED - len(features)))
    elif len(features) > EXPECTED:
        features = features[:EXPECTED]
    return features

def predict_intrusion(features):
    features = normalize_features(features)

    x = np.array(features).reshape(1, -1)
    x = scaler.transform(x)

    predictions = {}

    for name, model in models.items():
        pred = model.predict(x)
        label = label_encoder.inverse_transform(pred)[0]
        predictions[name] = label

    majority_vote = Counter(predictions.values()).most_common(1)[0][0]

    weighted_scores = {}
    for model_name, label in predictions.items():
        weighted_scores[label] = weighted_scores.get(label, 0) + accuracies[model_name]

    weighted_vote = max(weighted_scores, key=weighted_scores.get)

    return {
        "individual_predictions": predictions,
        "majority_voting": majority_vote,
        "weighted_voting": weighted_vote,
        "features_used": EXPECTED
    }
