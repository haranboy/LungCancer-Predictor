import google.generativeai as genai
import streamlit as st
import json

genai.configure(api_key="AIzaSyD7pJ9Kbkzl8U0PkjMVJ6YxWABnxnoiGis")  # Replace with your Gemini API Key

def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

st.title("Lung Cancer Predictor")

# Collect user input
user_inputs = {}
fields = ["Age", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy", "Wheezing", "Alcohol", "Coughing", "Breath Shortness", "Swallowing Difficulty", "Chest Pain"]

for field in fields:
    user_inputs[field] = st.selectbox(f"Do you have {field.lower()}?", ["Yes", "No"])

if st.button("Predict"):  
    prompt = f"Predict lung cancer probability based on: {json.dumps(user_inputs)}"
    result = get_gemini_response(prompt)
    st.write("Predicted Probability:", result)
