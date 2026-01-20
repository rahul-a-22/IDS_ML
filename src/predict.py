import pickle
import numpy as np
import os
from collections import defaultdict
from src.cl_kmeans_detect import detect_cl_kmeans

MODEL_PATH = os.path.join("models", "ids_multi_model.pkl")
CONFIDENCE_THRESHOLD = 0.6

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

models = artifacts["models"]
accuracies = artifacts["accuracies"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

EXPECTED_FEATURES = len(feature_names)
CLASS_NAMES = list(label_encoder.classes_)


def normalize_features(features):
    features = list(map(float, features))
    if len(features) < EXPECTED_FEATURES:
        features.extend([0.0] * (EXPECTED_FEATURES - len(features)))
    elif len(features) > EXPECTED_FEATURES:
        features = features[:EXPECTED_FEATURES]
    return features


def predict_intrusion(features):
    features = normalize_features(features)
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    individual_predictions = {}
    weighted_votes = defaultdict(float)

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
        else:
            pred_idx = model.predict(X_scaled)[0]
            probs = np.zeros(len(CLASS_NAMES))
            probs[pred_idx] = 1.0
            confidence = 1.0

        label = label_encoder.inverse_transform([pred_idx])[0]

        individual_predictions[name] = {
            "label": label,
            "confidence": round(confidence, 4)
        }

        for i, cls in enumerate(CLASS_NAMES):
            weighted_votes[cls] += probs[i] * accuracies[name]

    total_weight = sum(accuracies.values())
    weighted_probs = {
        cls: round(score / total_weight, 4)
        for cls, score in weighted_votes.items()
    }

    final_label = max(weighted_probs, key=weighted_probs.get)
    final_confidence = weighted_probs[final_label]

    if final_confidence < CONFIDENCE_THRESHOLD:
        cl_result = detect_cl_kmeans(features)

        if cl_result["is_anomaly"]:
            return {
                "type": "anomaly",
                "final_decision": {
                    "label": "ZERO_DAY_ATTACK",
                    "confidence": round(cl_result["distance"], 4)
                },
                "cluster_info": cl_result,
                "individual_predictions": individual_predictions,
                "weighted_voting": {
                    "label": final_label,
                    "confidence": final_confidence,
                    "probabilities": weighted_probs
                }
            }


    return {
        "type": "known",
        "final_decision": {
            "label": final_label,
            "confidence": final_confidence
        },
        "individual_predictions": individual_predictions,
        "weighted_voting": {
            "label": final_label,
            "confidence": final_confidence,
            "probabilities": weighted_probs
        }
    }
