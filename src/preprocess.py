import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(csv_path):
    """
    Loads dataset, cleans data, normalizes features,
    and encodes labels.
    """

    df = pd.read_csv(csv_path)

    # Replace missing & infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Separate features and labels
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Feature normalization (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled, y_encoded, scaler, label_encoder, X.columns
