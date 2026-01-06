# IDS_ML

> Lightweight Intrusion Detection System (IDS) using an ensemble of scikit-learn and XGBoost models.

## Summary
A compact toolkit to train and serve an ensemble-based intrusion detection model. Includes training scripts, a prediction helper, a Flask API, and a Streamlit dashboard for quick testing and visualization.

---

## Quickstart
1. Clone and prepare a virtual environment
```bash
git clone <repo-url>
cd IDS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the models
```bash
python src/train.py
```
This creates `models/ids_multi_model.pkl` (models, scaler, label encoder, and metrics).

3. Run the API
```bash
python api/app.py
```
Send POST requests to `http://127.0.0.1:5000/predict` with JSON `{ "features": [ ... ] }`.

4. Open the dashboard (optional)
```bash
streamlit run ui/dashboard.py
```
The dashboard interacts with the running API to display predictions.

5. Test on Terminal
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @input.json

```
---

## Usage example (Python)
```python
from src.predict import predict_intrusion
features = [0.0, 1.2, ...]  # same feature order used in training
print(predict_intrusion(features))
```

---

## Project layout
- `src/` — preprocessing, training and prediction utilities (`preprocess.py`, `train.py`, `predict.py`)
- `api/` — lightweight Flask server (`app.py`)
- `ui/` — Streamlit dashboard (`dashboard.py`)
- `data/` — example CSV datasets
- `models/` — saved model artifacts

---

## Notes
- Use Python 3.9+ and a virtual environment.
- Consider moving large model files to an external registry for production.

---

## Contributing 
Contributions welcome — please open issues or PRs. 
---
