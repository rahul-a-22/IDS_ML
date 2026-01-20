import pandas as pd
import streamlit as st
import requests
import pickle

API_URL = "http://127.0.0.1:5000/predict"

with open("models/ids_multi_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

feature_names = artifacts["feature_names"]
EXPECTED = len(feature_names)

st.set_page_config(page_title="IDS Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System")

st.subheader("Expected Features (Order Matters)")
st.write(", ".join(feature_names))

st.divider()
st.subheader("Input Features")

input_text = st.text_area(
    f"Enter comma-separated feature values (up to {EXPECTED})",
    placeholder="e.g. 12, 0.4, 89, 1024, 0, 0.3"
)

def parse_input(text):
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def label_color(label):
    if label == "BENIGN":
        return "#00c853"
    if label == "ZERO_DAY_ATTACK":
        return "#d50000"
    return "#ff1744"

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

            # ---------------- Individual Models ----------------
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

            # ---------------- Final Decision ----------------
            st.divider()
            st.subheader("Final Decision")

            decision = result["final_decision"]

            st.markdown(
                f"""
                <h1 style="color:{label_color(decision['label'])};">
                    {decision['label']}
                </h1>
                <h4 style="color:{confidence_color(decision['confidence'])};">
                    Confidence: {decision['confidence']}
                </h4>
                """,
                unsafe_allow_html=True
            )

            if result["type"] == "anomaly":
                st.error("⚠️ Zero-Day / Unknown Attack Detected")

            # ---------------- Weighted Voting ----------------
            st.divider()
            st.subheader("Weighted Voting Probabilities")

            probs = result["weighted_voting"]["probabilities"]

            df_probs = pd.DataFrame({
                "Class": list(probs.keys()),
                "Probability": list(probs.values())
            }).sort_values("Probability", ascending=False)

            st.bar_chart(df_probs.set_index("Class"))

            # ---------------- CL-KMeans Visualization ----------------
            if result["type"] == "anomaly" and "cluster_info" in result:
                st.divider()
                st.subheader("🚨 Cluster Distance Analysis (Zero-Day Detection)")

                cluster_info = result["cluster_info"]

                st.markdown(
                    f"""
                    **Assigned Cluster:** {cluster_info['cluster']}  
                    **Distance to Cluster Center:** {cluster_info['distance']:.4f}  
                    **Anomaly Threshold:** {cluster_info['threshold']:.4f}
                    """
                )

                df_dist = pd.DataFrame({
                    "Cluster": list(range(len(cluster_info["all_distances"]))),
                    "Distance": cluster_info["all_distances"]
                })

                st.bar_chart(df_dist.set_index("Cluster"))

    except Exception as e:
        st.error(f"Invalid input: {e}")
