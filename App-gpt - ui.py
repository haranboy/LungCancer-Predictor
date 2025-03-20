import streamlit as st
import google.generativeai as genai
import pickle
import pandas as pd
import os

def load_model():
    try:
        with open("lung_cancer_rf.pckl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: Model file 'lung_cancer_rf.pckl' not found. Please upload the model file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

def configure_api():
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        st.sidebar.warning("Gemini AI features are unavailable. Set the GEMINI_API_KEY environment variable.")
        return None
    genai.configure(api_key=API_KEY)
    return API_KEY

# Load Model & API
model = load_model()
API_KEY = configure_api()
feature_order = [
    'AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
    'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL',
    'IMMUNE_WEAKNESS', 'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION',
    'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'CHEST_TIGHTNESS',
    'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE'
]

# Sidebar UI
st.sidebar.image("https://via.placeholder.com/150", caption="Lung Health AI", use_column_width=True)
st.sidebar.header("Navigation")
st.sidebar.markdown("Use this tool to assess lung cancer risk.")

# Main UI
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
    response = st.radio(f"{feature.replace('_', ' ').title()}?", ('No', 'Yes'), horizontal=True)
    response_dict[feature] = 1 if response == 'Yes' else 0

# Numeric Inputs
response_dict["AGE"] = st.slider("Age:", 18, 100, 40)
response_dict["OXYGEN_SATURATION"] = st.slider("Oxygen Saturation (%):", 70, 100, 98)
response_dict["ENERGY_LEVEL"] = st.slider("Energy Level (1-10):", 1, 10, 5)

# Gender Selection (0 for Male, 1 for Female)
response_dict["GENDER"] = 1 if st.radio("Gender:", ('Male', 'Female'), horizontal=True) == 'Female' else 0

# Arrange responses in the exact feature order
user_responses = [response_dict[feature] for feature in feature_order]

# Prediction Button
if st.button("Predict Lung Cancer Risk"):
    with st.spinner("Analyzing risk factors..."):
        try:
            user_input_df = pd.DataFrame([user_responses], columns=feature_order)
            prediction_prob = model.predict_proba(user_input_df)[0][1] * 100

            # Color-coded result
            st.subheader("Predicted Lung Cancer Probability")
            if prediction_prob < 30:
                st.success(f"✅ Low Risk: {prediction_prob:.2f}%")
            elif prediction_prob < 70:
                st.warning(f"⚠️ Moderate Risk: {prediction_prob:.2f}%")
            else:
                st.error(f"🚨 High Risk: {prediction_prob:.2f}%")

            # AI Health Advice
            if API_KEY:
                try:
                    prompt = f"""
                    The patient's predicted lung cancer probability is {prediction_prob:.2f}%. 
                    Provide **specific medical advice and lifestyle changes** to help reduce the risk. 
                    Keep it concise and actionable.
                    """
                    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                    response = model_gemini.generate_content(prompt)
                    st.subheader("AI Health Advice")
                    with st.expander("Click to view AI-generated health advice"):
                        st.write(response.text)
                except Exception as e:
                    st.warning(f"Gemini API Error: {e}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
