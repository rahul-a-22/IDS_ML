# IDS_ML

> 🚨 Intrusion Detection System (IDS) using ensemble voting of scikit-learn + XGBoost models.

## 🔍 Overview

**IDS_ML** trains several classifiers on network traffic features and combines their predictions via majority and weighted voting to detect intrusions. The repo includes scripts for training (`src/train.py`), inference (`src/predict.py`), a Flask API (`api/app.py`), and a Streamlit dashboard (`ui/dashboard.py`).

---

## ✅ Features

- Multiple classifiers: RandomForest, ExtraTrees, DecisionTree, XGBoost
- Standardized preprocessing and label encoding
- Majority and weighted voting aggregation
- REST API for predictions and a Streamlit dashboard for interaction

---

## 📁 Repository Structure

```
.
├─ api/
│  └─ app.py            # Flask API for model predictions
├─ data/                # Sample datasets (CSV)
├─ models/              # Trained model artifacts (ids_multi_model.pkl)
├─ src/
│  ├─ train.py          # Train models and save artifacts
│  └─ predict.py        # Prediction helper (used by API & dashboard)
├─ ui/
│  └─ dashboard.py      # Streamlit dashboard
├─ requirements.txt
├─ input.json           # Example input for API or testing
└─ README.md
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+ recommended
- Git
- (Optional) Create a virtual environment

### Setup
```bash
# create & activate venv (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Train models
```bash
python src/train.py
```
- Outputs: `models/ids_multi_model.pkl` (contains models, scaler, label encoder, accuracies)

### Predict (programmatically)
```python
from src.predict import predict_intrusion

# features must be a list/array with the same length as the training features
features = [0.0, 1.2, ...]
result = predict_intrusion(features)
print(result)
```

### Start the API
```bash
python api/app.py
```
- POST JSON to `http://127.0.0.1:5000/predict` with body: `{ "features": [ ... ] }`

### Launch the dashboard
```bash
streamlit run ui/dashboard.py
```
- Ensure the API is running; the dashboard sends requests to `/predict`.

---

## 💡 Tips & Notes
- The `models/ids_multi_model.pkl` file is included by default in this repo, but consider using large-file storage or external model registry for bigger artifacts.
- `src/train.py` prints and stores model accuracies in the pickle for use in the UI.

---

## ⚠️ Recommended `.gitignore`

```
# Python
venv/
__pycache__/
*.pyc

# Notebook
.ipynb_checkpoints/

# Models & data
models/*.pkl
data/

# OS / IDE
.DS_Store
*.log

# Environment
.env
```

---

## 🤝 Contributing
PRs and issues are welcome. Please add tests or small incremental changes and follow clear commit messages.

## 📄 License
Add a license file to clarify usage (e.g., MIT).

---

If you want, I can also create the `.gitignore` file now and stage these edits for commit — shall I proceed?