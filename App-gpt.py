import streamlit as st
import google.generativeai as genai
import os
import pickle
import pandas as pd
import sklearn
import numpy

# Configure Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=API_KEY)

    # Load the Random Forest model
    try:
        with open("lung_cancer_rf.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("lung_cancer_rf.pckl not found. Please ensure it's in the same directory.")
        exit()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        exit()

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

    # Yes/No questions
    yes_no_features = [
        "SMOKING", "THROAT_DISCOMFORT", "BREATHING_ISSUE", "SMOKING_FAMILY_HISTORY",
        "STRESS_IMMUNE", "EXPOSURE_TO_POLLUTION", "FAMILY_HISTORY", "IMMUNE_WEAKNESS",
        "CHEST_TIGHTNESS", "ALCOHOL_CONSUMPTION", "LONG_TERM_ILLNESS", "MENTAL_STRESS",
        "FINGER_DISCOLORATION"
    ]

    for feature in yes_no_features:
        response = st.radio(f"Do you have {feature.replace('_', ' ').lower()}?", ('No', 'Yes'))
        user_responses.append(1 if response == 'Yes' else 0)

    # Numeric inputs
    age = st.slider("What is your age?", min_value=18, max_value=100, value=40)
    oxygen_saturation = st.slider("Oxygen saturation level (%)", min_value=70, max_value=100, value=98)
    energy_level = st.slider("Rate your energy level (1-10)", min_value=1, max_value=10, value=5)

    user_responses.append(age)
    user_responses.append(oxygen_saturation)
    user_responses.append(energy_level)

    # Gender selection (0 for Male, 1 for Female)
    gender = st.radio("Select your gender:", ('Male', 'Female'))
    user_responses.append(1 if gender == 'Female' else 0)

    # Prediction using Random Forest model
    if st.button("Predict"):
        try:
            user_input_df = pd.DataFrame([user_responses], columns=feature_names)
            prediction = model.predict(user_input_df)  # Predict class label.
            user_input_str = ", ".join([f"{feature}: {value}" for feature, value in zip(feature_names, user_responses)])

            # Construct prompt for Gemini (modified)
            prompt = f"""
            The patient's predicted lung cancer risk is: {prediction}. Based on these factors: {user_input_str}.
            Based on these factors, how can the patient stay healthier for longer? Provide specific lifestyle changes and medical advice.
            Include helpful links.
            """

            model_gemini = genai.GenerativeModel("gemini-pro")
            response = model_gemini.generate_content(prompt)

            st.subheader("Lung Cancer Risk and Advice:")
            st.write(response.text)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
