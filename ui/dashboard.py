import streamlit as st
import requests
import pickle

API_URL = "http://127.0.0.1:5000/predict"

with open("models/ids_multi_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

feature_names = artifacts["feature_names"]
EXPECTED = len(feature_names)
accuracies = artifacts["accuracies"]

st.set_page_config(page_title="IDS Voting Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System – Voting Dashboard")

st.subheader("Model Accuracies")
for model, acc in accuracies.items():
    st.write(f"{model}: {acc*100:.4f}%")

st.divider()
st.subheader("Input Network Traffic Features")

st.subheader("Expected Features")
st.write(", ".join(feature_names))

st.divider()
st.subheader("Enter Feature Values")

input_text = st.text_area(
    "Comma-separated feature values",
    placeholder="e.g. 12, 0.4, 89, 1024, 0, 0.3 ..."
)

def parse_input(text):
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]

if st.button("Predict"):
    try:
        features = parse_input(input_text)

        response = requests.post(
            API_URL,
            json={"features": features},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"Features used: {result['features_used']}")

            st.subheader("Individual Model Predictions")
            for model, pred in result["individual_predictions"].items():
                st.write(f"{model}: {pred}")

            def color_label(label):
                if label.upper() == "BENIGN":
                    return f"<h2 style='color: #00c853'>{label}</h2>"
                return f"<h2 style='color: #ff1744'>{label}</h2>"

            st.divider()
            st.subheader("Final Decision")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Majority Voting**", unsafe_allow_html=True)
                st.markdown(color_label(result["majority_voting"]), unsafe_allow_html=True)

            with col2:
                st.markdown("**Weighted Voting**", unsafe_allow_html=True)
                st.markdown(color_label(result["weighted_voting"]), unsafe_allow_html=True)

        else:
            st.error(response.json())

    except Exception as e:
        st.error(f"Connection error: {e}")


