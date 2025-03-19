import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import sklearn

# Load API Key from Streamlit Secrets
api_key = st.secrets["GEMINI_API_KEY"]

# Configure Gemini API
genai.configure(api_key=api_key)

# Load the trained Random Forest model
MODEL_PATH = "lung_cancer_rf.pckl"  # Ensure the file is in the same directory

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_PATH}' not found! Ensure it's uploaded.")
    st.stop()

# 🌟 Streamlit UI
st.title("🔬 Lung Cancer Predictor (ML Model)")

# Define input fields
fields = [
    "Age", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease",
    "Fatigue", "Allergy", "Wheezing", "Alcohol", "Coughing", "Breath Shortness",
    "Swallowing Difficulty", "Chest Pain"
]

user_inputs = {}

# Collect user inputs
for field in fields:
    user_inputs[field] = st.selectbox(f"Do you have {field.lower()}?", ["Yes", "No"])

# Convert categorical inputs (Yes → 1, No → 0)
input_values = np.array([1 if user_inputs[field] == "Yes" else 0 for field in fields]).reshape(1, -1)

# 🎯 Make Prediction
if st.button("🔍 Predict"):
    try:
        probability = model.predict_proba(input_values)[0][1] * 100  # Get probability of "Yes"
        st.success(f"📊 Predicted Lung Cancer Risk: **{probability:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
