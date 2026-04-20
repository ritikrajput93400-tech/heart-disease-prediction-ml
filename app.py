import streamlit as st
import pandas as pd
import joblib

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ------------------ Load Files ------------------
model = joblib.load("KNN_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ------------------ Title ------------------
st.title("❤️ Heart Disease Prediction App")
st.caption("Machine Learning based prediction | By Ritik Rajput")

st.markdown("---")

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ------------------ Predict Button ------------------
if st.button("🔍 Predict Heart Disease Risk"):

    # Raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # ------------------ Output ------------------
    st.markdown("### 🩺 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.info(f"📊 Risk Probability: **{probability * 100:.2f}%**")
