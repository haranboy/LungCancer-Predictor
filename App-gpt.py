import streamlit as st
import joblib
import numpy as np
import openai

# Load trained Random Forest model
rf_model = joblib.load("lung_cancer_rf.pckl")

# Set OpenAI API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Feature names
feature_names = [
    "SMOKING", "ENERGY_LEVEL", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "OXYGEN_SATURATION",
    "AGE", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE", "EXPOSURE_TO_POLLUTION", "FAMILY_HISTORY",
    "IMMUNE_WEAKNESS", "CHEST_TIGHTNESS", "ALCOHOL_CONSUMPTION", "LONG_TERM_ILLNESS",
    "MENTAL_STRESS", "GENDER", "FINGER_DISCOLORATION"
]

# Streamlit UI
st.title("Lung Cancer Prediction App")
st.write("Answer the following questions to assess your lung cancer risk.")

# User inputs
user_responses = []

# Yes/No Questions
yes_no_features = [
    "SMOKING", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "SMOKING_FAMILY_HISTORY", 
    "STRESS_IMMUNE", "EXPOSURE_TO_POLLUTION", "FAMILY_HISTORY", "IMMUNE_WEAKNESS", 
    "CHEST_TIGHTNESS", "ALCOHOL_CONSUMPTION", "LONG_TERM_ILLNESS", "MENTAL_STRESS", 
    "FINGER_DISCOLORATION"
]

for feature in yes_no_features:
    response = st.radio(f"Do you have {feature.replace('_', ' ').lower()}?", ('No', 'Yes'))
    user_responses.append(1 if response == 'Yes' else 0)

# Numeric Inputs
age = st.slider("What is your age?", min_value=18, max_value=100, value=40)
oxygen_saturation = st.slider("Oxygen saturation level (%)", min_value=70, max_value=100, value=98)
energy_level = st.slider("Rate your energy level (1-10)", min_value=1, max_value=10, value=5)

user_responses.append(age)
user_responses.append(oxygen_saturation)
user_responses.append(energy_level)

# Gender Selection
gender = st.radio("Select your gender:", ('Male', 'Female'))
user_responses.append(1 if gender == 'Female' else 0)

# Prediction
if st.button("Predict"):
    input_array = np.array([user_responses]).reshape(1, -1)
    probability = rf_model.predict_proba(input_array)[0][1] * 100  

    # Display Probability
    st.subheader(f"Lung Cancer Probability: {probability:.2f}%")

    # Generate AI Explanation
    prompt = f"""
    The user has a lung cancer probability of {probability:.2f}%. 
    They provided the following answers:
    {dict(zip(feature_names, user_responses))}
    
    Provide an easy-to-understand explanation of what this result means. 
    Also, suggest preventive measures or lifestyle changes based on their risk factors.
    """
    
    # OpenAI API Call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Using GPT-3.5-turbo instead of GPT-4
        messages=[
            {"role": "system", "content": "You are a medical assistant providing clear, simple advice."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_explanation = response["choices"][0]["message"]["content"]
    
    st.write("### AI Health Advice:")
    st.info(gpt_explanation)
