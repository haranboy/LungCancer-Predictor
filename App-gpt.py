import google.generativeai as genai
import streamlit as st
import json
import os

# 🔒 Secure API Key (set via environment variable)
api_key = os.getenv("GEMINI_API_KEY")  # Set this in your system or Streamlit secrets
if not api_key:
    st.error("❌ API Key missing! Set GEMINI_API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=api_key)

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")  # ✅ Use latest version
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# 🌟 Streamlit UI
st.title("🔬 Lung Cancer Predictor")

# User Input Fields
user_inputs = {}
fields = [
    "Age", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease",
    "Fatigue", "Allergy", "Wheezing", "Alcohol", "Coughing", "Breath Shortness",
    "Swallowing Difficulty", "Chest Pain"
]

for field in fields:
    user_inputs[field] = st.selectbox(f"Do you have {field.lower()}?", ["Yes", "No"])

# 🎯 Make Prediction
if st.button("🔍 Predict"):  
    prompt = (
        "Given the following patient symptoms, predict the lung cancer probability "
        "as a percentage (e.g., 'Risk: 75%'). Patient details: "
        f"{json.dumps(user_inputs)}"
    )
    
    result = get_gemini_response(prompt)

    if "Error" in result:
        st.error(result)
    else:
        st.success(f"📊 Predicted Probability: {result}")
