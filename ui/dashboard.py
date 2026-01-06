import pandas as pd
import streamlit as st
import requests
import pickle

API_URL = "http://127.0.0.1:5000/predict"

with open("models/ids_multi_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

feature_names = artifacts["feature_names"]

st.set_page_config(page_title="IDS Voting Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System")

st.subheader("Expected Features (Order Matters)")
st.write(", ".join(feature_names))

st.divider()
st.subheader("Input Features")

input_text = st.text_area(
    "Enter comma-separated feature values",
    placeholder="e.g. 12, 0.4, 89, 1024, 0, 0.3"
)

def parse_input(text):
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def label_color(label):
    return "#00c853" if label.upper() == "BENIGN" else "#ff1744"

def confidence_color(conf):
    if conf >= 0.8:
        return "#00c853"
    elif conf >= 0.5:
        return "#ffab00"
    return "#ff1744"

if st.button("Predict"):
    try:
        features = parse_input(input_text)

        response = requests.post(
            API_URL,
            json={"features": features},
            timeout=10
        )

        if response.status_code != 200:
            st.error(response.json())
        else:
            result = response.json()

            st.divider()
            st.subheader("Individual Model Predictions")

            for model, data in result["individual_predictions"].items():
                st.markdown(
                    f"""
                    <b>{model}</b><br>
                    <span style="color:{label_color(data['label'])}; font-size:22px;">
                        {data['label']}
                    </span><br>
                    <span style="color:{confidence_color(data['confidence'])};">
                        Confidence: {data['confidence']}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()
            st.subheader("Final Decision")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Majority Voting**")
                if "majority_voting" in result:
                    label = result["majority_voting"]["label"]
                else:
                    label = result["weighted_voting"]["label"]

                st.markdown(
                    f"""<h2 style='color:{label_color(label)};'>{label}</h2>""",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("**Weighted Voting**")
                st.markdown(
                    f"""
                    <h2 style="color:{label_color(result['weighted_voting']['label'])};">
                        {result['weighted_voting']['label']}
                    </h2>
                    <span style="color:{confidence_color(result['weighted_voting']['confidence'])};">
                        Confidence: {result['weighted_voting']['confidence']}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

            if "probabilities" in result["weighted_voting"]:
                st.divider()
                st.subheader("Prediction Probability Distribution")

                probs = result["weighted_voting"]["probabilities"]

                df_probs = pd.DataFrame({
                    "Class": list(probs.keys()),
                    "Probability": list(probs.values())
                }).sort_values("Probability", ascending=False)

                st.bar_chart(df_probs.set_index("Class"))

    except Exception as e:
        st.error(f"Invalid input: {e}")
