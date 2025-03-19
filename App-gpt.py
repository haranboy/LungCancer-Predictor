import streamlit as st
import google.generativeai as genai
import pickle
import pandas as pd
import os

# Secure API Key Handling
API_KEY = os.getenv("GEMINI_API_KEY")  # For local testing
# API_KEY = st.secrets["GEMINI_API_KEY"]  # Uncomment for secure deployment

if not API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=API_KEY)

# Load the Random Forest Model
try:
    with open("lung_cancer_rf.pckl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model file 'lung_cancer_rf.pckl' not found. Please upload the model file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define Feature Names (Must match training data exactly)
feature_names = [
    "SMOKING", "ENERGY_LEVEL", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "OXYGEN_SATURATION",
    "AGE", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE", "EXPOSURE_TO_POLLUTION", "FAMILY_HISTORY",
    "IMMUNE_WEAKNESS", "CHEST_TIGHTNESS", "ALCOHOL_CONSUMPTION", "LONG_TERM_ILLNESS",
    "MENTAL_STRESS", "GENDER", "FINGER_DISCOLORATION"
]

# Streamlit UI
st.title("Lung Cancer Prediction App")
st.write("Answer the following questions to assess your lung cancer risk.")

# User Inputs Storage
response_dict = {feature: 0 for feature in feature_names}  # Default all responses to 0

# Yes/No Questions
yes_no_features = [
    "SMOKING", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE", "EXPOSURE_TO_POLLUTION", "FAMILY_HISTORY", "IMMUNE_WEAKNESS",
    "CHEST_TIGHTNESS", "ALCOHOL_CONSUMPTION", "LONG_TERM_ILLNESS", "MENTAL_STRESS",
    "FINGER_DISCOLORATION"
]

for feature in yes_no_features:
    response = st.radio(f"Do you have {feature.replace('_', ' ').lower()}?", ('No', 'Yes'))
    response_dict[feature] = 1 if response == 'Yes' else 0

# Numeric Inputs
response_dict["AGE"] = st.slider("What is your age?", min_value=18, max_value=100, value=40)
response_dict["OXYGEN_SATURATION"] = st.slider("Oxygen saturation level (%)", min_value=70, max_value=100, value=98)
response_dict["ENERGY_LEVEL"] = st.slider("Rate your energy level (1-10)", min_value=1, max_value=10, value=5)

# Gender Selection (0 for Male, 1 for Female)
response_dict["GENDER"] = 1 if st.radio("Select your gender:", ('Male', 'Female')) == 'Female' else 0

# Convert Dictionary to Ordered List
user_responses = [response_dict[feature] for feature in feature_names]

# Prediction Button
if st.button("Predict"):
    try:
        # Convert input to DataFrame with correct feature order
        user_input_df = pd.DataFrame([user_responses], columns=feature_names)

        # Predict Probability
        prediction_prob = model.predict_proba(user_input_df)[0][1] * 100  # Convert to percentage

        # Display Prediction
        st.subheader("Predicted Lung Cancer Probability:")
        st.write(f"⚠️ **{prediction_prob:.2f}% chance of lung cancer**")

        # If API is set, Generate AI Response
        if API_KEY:
            try:
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
                st.warning(f"Gemini API Error: {e}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
