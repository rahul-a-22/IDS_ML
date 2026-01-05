import streamlit as st
import requests
import pickle

API_URL = "http://127.0.0.1:5000/predict"

with open("models/ids_multi_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

feature_names = artifacts["feature_names"]
accuracies = artifacts["accuracies"]

st.set_page_config(page_title="IDS Voting Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System – Voting Dashboard")

st.subheader("Model Accuracies")
for model, acc in accuracies.items():
    st.write(f"{model}: {acc*100:.4f}%")

st.divider()
st.subheader("Input Network Traffic Features")

features = []
for name in feature_names:
    features.append(st.number_input(name, value=0.0))

if st.button("Predict"):
    payload = {"features": features}

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.divider()
            st.subheader("Individual Model Predictions")

            for model, pred in result["individual_predictions"].items():
                st.write(f"{model}: {pred}")
                
            def color_label(label):
                if label.upper() == "BENIGN":
                    return f"<h2 style='color: #00c853;'>{label}</h2>"
                else:
                    return f"<h2 style='color: #ff1744;'>{label}</h2>"

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
