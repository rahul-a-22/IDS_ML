# Intrusion Detection System using Ensemble Learning

This project implements a lightweight yet effective Intrusion Detection System (IDS) that leverages an ensemble of machine learning models to identify network intrusions. It includes scripts for training, a Flask API for serving predictions, and a Streamlit dashboard for real-time interaction and visualization.

## Features

- **Ensemble-Based Model**: Combines predictions from multiple models (RandomForest, ExtraTrees, DecisionTree, and XGBoost) for higher accuracy and robustness.
- **Weighted Voting**: Uses a weighted voting mechanism based on model accuracies to make final predictions.
- **REST API**: A simple Flask API to serve predictions.
- **Interactive Dashboard**: A Streamlit dashboard to visualize predictions and model performance.
- **Batch Prediction**: A script to make predictions on a large dataset from a CSV file.
- **Modular Design**: The project is structured into separate modules for training, prediction, and serving, making it easy to extend and maintain.

## System Architecture

The system is composed of three main components:

1.  **Training Pipeline**: The `src/train.py` script loads the training data, preprocesses it, trains the models, and saves the trained models and artifacts to a pickle file.
2.  **Flask API**: The `api/app.py` script loads the saved models and provides a `/predict` endpoint to make predictions on new data.
3.  **Streamlit Dashboard**: The `ui/dashboard.py` script provides a user-friendly interface to interact with the API and visualize the predictions.

```
+-----------------+      +-----------------+      +---------------------+
|                 |----->|                 |----->|                     |
| Training        |      | Model Artifacts |      | Flask API           |
| (src/train.py)  |      | (models/*.pkl)  |      | (api/app.py)        |
|                 |<-----|                 |<-----|                     |
+-----------------+      +-----------------+      +----------+----------+
                                                              |
                                                              |
                                                     +--------v--------+
                                                     |                 |
                                                     | Streamlit       |
                                                     | Dashboard       |
                                                     | (ui/dashboard.py)|
                                                     |                 |
                                                     +-----------------+
```

## Getting Started

### Prerequisites

- Python 3.9+
- A virtual environment tool (e.g., `venv`)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/IDS_ML.git
    cd IDS_ML
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Model Training

To train the models, run the following command:

```bash
python src/train.py
```

This will create a `models/ids_multi_model.pkl` file containing the trained models, scaler, label encoder, and feature names.

### 2. Running the Application

To run the Flask API and the Streamlit dashboard simultaneously, use the `run_app.py` script:

```bash
python run_app.py
```

This will start the Flask API on `http://127.0.0.1:5000` and the Streamlit dashboard on `http://localhost:8501`.

### 3. API

The Flask API has a single endpoint:

- **POST /predict**

  This endpoint accepts a JSON object with a `features` key, which should be a list of numerical feature values.

  **Example Request:**

  ```bash
  curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [0.0, 1.2, ...]}'
  ```

  **Example Response:**

  ```json
  {
    "individual_predictions": {
      "RandomForest": {
        "label": "BENIGN",
        "confidence": 0.98
      },
      ...
    },
    "weighted_voting": {
      "label": "BENIGN",
      "confidence": 0.99,
      "probabilities": {
        "BENIGN": 0.99,
        "ATTACK": 0.01
      }
    }
  }
  ```

### 4. Dashboard

The Streamlit dashboard provides a user-friendly interface to interact with the API. You can enter the feature values in a text area and get the predictions in real-time.

### 5. Batch Prediction

To make predictions on a batch of data from a CSV file, use the `src/batch_predict.py` script.

```bash
python src/batch_predict.py
```

This script will read the data from `data/test.csv`, make predictions, and save the results to `data/test_output.csv`.

## Project Structure

```
.
├── api
│   └── app.py              # Flask API for serving predictions
├── data
│   ├── CICIDS2017_sample.csv # Sample training data
│   └── test.csv            # Sample batch prediction data
├── models
│   └── ids_multi_model.pkl # Saved model artifacts
├── src
│   ├── batch_predict.py    # Script for batch prediction
│   ├── predict.py          # Prediction logic
│   ├── preprocess.py       # Data preprocessing utilities
│   └── train.py            # Model training script
├── ui
│   └── dashboard.py        # Streamlit dashboard
├── .gitignore
├── input.json
├── MTH_IDS_IoTJ.ipynb
├── README.md
├── requirements.txt
└── run_app.py              # Main script to run the application
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.