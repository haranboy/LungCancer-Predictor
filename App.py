import streamlit as st
import joblib
import openai  # GPT API
import pandas as pd

# Load the trained model
model = joblib.load("model.pckl")

# Set up OpenAI API
openai.api_key = "your_openai_api_key"  # Replace with your key

# Function to get GPT explanation
def get_gpt_explanation(user_inputs, prediction):
    prompt = f"""
    A patient has entered the following symptoms: {user_inputs}.
    The AI model predicted a {prediction}% chance of lung cancer.
    Explain why this prediction was made and suggest lifestyle changes.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a medical assistant providing helpful advice."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("Lung Cancer Prediction & AI Advisor")

# Collect user inputs
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
age = st.number_input("Enter your age", min_value=10, max_value=100)
breathing_issue = st.selectbox("Do you have breathing issues?", ["Yes", "No"])

# Convert to DataFrame for model
input_data = pd.DataFrame([[smoking, age, breathing_issue]], 
                          columns=['SMOKING', 'AGE', 'BREATHING_ISSUE'])

# Convert categorical to numeric
input_data.replace({'Yes': 1, 'No': 0}, inplace=True)

# Make Prediction
prediction_proba = model.predict_proba(input_data)[0][1] * 100  # Get percentage
st.write(f"🩺 Your lung cancer risk: **{prediction_proba:.2f}%**")

# Get GPT Explanation
if st.button("Ask AI for Advice"):
    advice = get_gpt_explanation(input_data.to_dict(), prediction_proba)
    st.write("💡 AI Advice:")
    st.write(advice)
