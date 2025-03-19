import streamlit as st
import google.generativeai as genai
import os
import pickle
import pandas as pd

# Configure Gemini API Key using Streamlit secrets (replace with st.secrets in deployment)
API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with st.secrets["GEMINI_API_KEY"] in a secure environment

if not API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=API_KEY)

# Load the Random Forest model
try:
    with open("lung_cancer_rf.pckl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: lung_cancer_rf.pckl not found. Please upload the model file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

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

# Yes/No questions
yes_no_features = [
    "SMOKING", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE", "EXPOSURE_TO_POLLUTION", "FAMILY_HISTORY", "IMMUNE_WEAKNESS",
    "CHEST_TIGHTNESS", "ALCOHOL_CONSUMPTION", "LONG_TERM_ILLNESS", "MENTAL_STRESS",
    "FINGER_DISCOLORATION"
]

# Create a dictionary to store responses, keyed by feature name
response_dict = {}

for feature in yes_no_features:
    response = st.radio(f"Do you have {feature.replace('_', ' ').lower()}?", ('No', 'Yes'))
    response_dict[feature] = 1 if response == 'Yes' else 0

# Numeric inputs
age = st.slider("What is your age?", min_value=18, max_value=100, value=40)
oxygen_saturation = st.slider("Oxygen saturation level (%)", min_value=70, max_value=100, value=98)
energy_level = st.slider("Rate your energy level (1-10)", min_value=1, max_value=10, value=5)

response_dict["AGE"] = age
response_dict["OXYGEN_SATURATION"] = oxygen_saturation
response_dict["ENERGY_LEVEL"] = energy_level

# Gender selection (0 for Male, 1 for Female)
gender = st.radio("Select your gender:", ('Male', 'Female'))
response_dict["GENDER"] = 1 if gender == 'Female' else 0

# Append user responses in the order of feature_names
for feature in feature_names:
    user_responses.append(response_dict[feature])

# Prediction using Random Forest model
if st.button("Predict"):
    try:
        # Ensure the input data matches the trained model’s feature order
        user_input_df = pd.DataFrame([user_responses], columns=feature_names)

        # Predict probability of lung cancer
        prediction_prob = model.predict_proba(user_input_df)[0][1] * 100  # Convert to percentage

        # Display prediction result
        st.subheader("Predicted Lung Cancer Probability:")
        st.write(f"⚠️ **{prediction_prob:.2f}% chance of lung cancer**")

        # Construct prompt for Gemini AI (Only if API is configured)
        if API_KEY:
            user_input_str = ", ".join([f"{feature}: {value}" for feature, value in zip(feature_names, user_responses)])

            prompt = f"""
            The patient's predicted lung cancer probability is {prediction_prob:.2f}%. 
            Factors considered: {user_input_str}. 

            Provide **specific medical advice and lifestyle changes** to help reduce the risk. 
            Keep it concise and actionable. Suggest **helpful resources or medical professionals to consult.**
            """

            model_gemini = genai.GenerativeModel("gemini-pro")
            response = model_gemini.generate_content(prompt)

            # Display AI-generated advice
            st.subheader("AI Health Advice:")
            st.write(response.text)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
