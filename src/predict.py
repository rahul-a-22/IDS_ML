import pickle
import numpy as np
import os
from collections import Counter, defaultdict

MODEL_PATH = os.path.join("models", "ids_multi_model.pkl")

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

models = artifacts["models"]
accuracies = artifacts["accuracies"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

EXPECTED = len(feature_names)
CLASS_NAMES = list(label_encoder.classes_)

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

    individual = {}
    prob_accumulator = defaultdict(float)

    for name, model in models.items():
        pred = model.predict(x)[0]
        label = label_encoder.inverse_transform([pred])[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
        else:
            probs = np.zeros(len(CLASS_NAMES))
            probs[pred] = 1.0

        confidence = float(max(probs))

        individual[name] = {
            "label": label,
            "confidence": round(confidence, 4)
        }

        for i, cls in enumerate(CLASS_NAMES):
            prob_accumulator[cls] += probs[i] * accuracies[name]

    total_weight = sum(accuracies.values())
    weighted_probs = {k: round(v / total_weight, 4) for k, v in prob_accumulator.items()}

    weighted_label = max(weighted_probs, key=weighted_probs.get)

    return {
        "individual_predictions": individual,
        "weighted_voting": {
            "label": weighted_label,
            "confidence": weighted_probs[weighted_label],
            "probabilities": weighted_probs
        }
    }
