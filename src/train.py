import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

df = pd.read_csv("data/CICIDS2017_sample.csv")

X_df = df.drop("Label", axis=1)
y = df["Label"]

X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_df.fillna(0, inplace=True)
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df.fillna(0, inplace=True)

feature_names = list(X_df.columns)

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X_df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=50, eval_metric="mlogloss")
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies[name] = acc
    print(f"{name}: {acc*100:.5f}%")

with open("models/ids_multi_model.pkl", "wb") as f:
    pickle.dump(
        {
            "models": models,
            "scaler": scaler,
            "label_encoder": le,
            "feature_names": feature_names,
            "accuracies": accuracies
        },
        f
    )
