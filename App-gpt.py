import streamlit as st
import google.generativeai as genai
import pickle
import pandas as pd
import os

# Secure API Key Handling
API_KEY = os.getenv("GEMINI_API_KEY")  # Use st.secrets["GEMINI_API_KEY"] in deployment

if not API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=API_KEY)

# Load the trained model
try:
    with open("lung_cancer_rf.pckl", "rb") as f:
        model = pickle.load(f)

    # Define the required feature order
    feature_order = [
        'AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
        'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL',
        'IMMUNE_WEAKNESS', 'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION',
        'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'CHEST_TIGHTNESS',
        'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE'
    ]

except FileNotFoundError:
    st.error("Error: Model file 'lung_cancer_rf.pckl' not found. Please upload the model file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Streamlit UI
st.title("Lung Cancer Prediction App")
st.write("Answer the following questions to assess your lung cancer risk.")

# User Inputs Storage
response_dict = {feature: 0 for feature in feature_order}  # Initialize with default values

# Yes/No Features
yes_no_features = [
    "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS", "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS", "IMMUNE_WEAKNESS", "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT", "CHEST_TIGHTNESS", "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE"
]

for feature in yes_no_features:
    response = st.radio(f"Do you have {feature.replace('_', ' ').lower()}?", ('No', 'Yes'))
    response_dict[feature] = 1 if response == 'Yes' else 0

# Numeric Inputs
numeric_inputs = {
    "AGE": st.slider("What is your age?", min_value=18, max_value=100, value=40),
    "OXYGEN_SATURATION": st.slider("Oxygen saturation level (%)", min_value=70, max_value=100, value=98),
    "ENERGY_LEVEL": st.slider("Rate your energy level (1-10)", min_value=1, max_value=10, value=5)
}

for feature, value in numeric_inputs.items():
    response_dict[feature] = value

# Gender Selection (0 for Male, 1 for Female)
response_dict["GENDER"] = 1 if st.radio("Select your gender:", ('Male', 'Female')) == 'Female' else 0

# Arrange responses in the exact feature order
user_responses = [response_dict[feature] for feature in feature_order]

# Prediction Button
if st.button("Predict"):
    try:
        # Create DataFrame in the correct feature order
        user_input_df = pd.DataFrame([user_responses], columns=feature_order)

        # Predict Probability
        prediction_prob = model.predict_proba(user_input_df)[0][1] * 100  # Convert to percentage

        # Display Prediction
        st.subheader("Predicted Lung Cancer Probability:")
        st.write(f"⚠️ **{prediction_prob:.2f}% chance of lung cancer**")

        # If API is set, Generate AI Response
        if API_KEY:
            try:
                user_input_str = ", ".join([f"{feature}: {value}" for feature, value in zip(feature_order, user_responses)])

                prompt = f"""
                The patient's predicted lung cancer probability is {prediction_prob:.2f}%. 
                Factors considered: {user_input_str}. 

                Provide **specific medical advice and lifestyle changes** to help reduce the risk. 
                Keep it concise and actionable. Suggest **helpful resources or medical professionals to consult.**
                """

                model_gemini = genai.GenerativeModel("gemini-1.0-pro")
                response = model_gemini.generate_content(prompt)

                # Display AI-generated advice
                st.subheader("AI Health Advice:")
                st.write(response.text)

            except Exception as e:
                st.warning(f"Gemini API Error: {e}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
