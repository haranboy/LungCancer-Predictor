import streamlit as st
import google.generativeai as genai
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

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
st.sidebar.image("https://via.placeholder.com/150", caption="Lung Health AI", use_container_width=True)
st.sidebar.header("Navigation")
st.sidebar.markdown("Use this tool to assess lung cancer risk.")

# Layout: Left (Inputs) | Right (Results)
col1, col2 = st.columns([2, 1])  # Left: 2x width, Right: 1x width

with col1:
    st.title("Lung Cancer Prediction")
    st.write("Answer the questions to assess your lung cancer risk.")

    response_dict = {feature: 0 for feature in feature_order}  # Initialize defaults

    # Grid Layout (3 per row)
    input_cols = st.columns(3)

    # Numeric Inputs (Age, Oxygen, Energy)
    response_dict["AGE"] = input_cols[0].slider("Age:", 18, 100, 40)
    response_dict["OXYGEN_SATURATION"] = input_cols[1].slider("Oxygen Saturation (%):", 70, 100, 98)
    response_dict["ENERGY_LEVEL"] = input_cols[2].slider("Energy Level (1-10):", 1, 10, 5)

    # Gender Selection
    response_dict["GENDER"] = 1 if input_cols[0].radio("Gender:", ('Male', 'Female'), horizontal=True) == 'Female' else 0

    # Yes/No Features in rows of 3
    yes_no_features = [
        "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS", "EXPOSURE_TO_POLLUTION",
        "LONG_TERM_ILLNESS", "IMMUNE_WEAKNESS", "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION",
        "THROAT_DISCOMFORT", "CHEST_TIGHTNESS", "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY",
        "STRESS_IMMUNE"
    ]
    
    for i in range(0, len(yes_no_features), 3):
        row = st.columns(3)
        for j, feature in enumerate(yes_no_features[i:i+3]):
            response_dict[feature] = 1 if row[j].radio(feature.replace("_", " ").title(), ('No', 'Yes'), horizontal=True) == 'Yes' else 0

# Predict Button on Right Side
with col2:
    if st.button("Predict", use_container_width=True):
        with st.spinner("Analyzing risk factors..."):
            try:
                user_input_df = pd.DataFrame([[response_dict[feature] for feature in feature_order]], columns=feature_order)
                prediction_prob = model.predict_proba(user_input_df)[0][1] * 100

                # Results Section
                st.subheader("Prediction Result")
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

                # Feature Importance Visualization
                st.subheader("🔍 Factors Affecting Your Risk")
                feature_importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_order,
                    'Importance': feature_importances
                }).sort_values(by="Importance", ascending=False)

                # Plot feature importance
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
                ax.set_xlabel("Importance Score")
                ax.set_title("Feature Importance")
                ax.invert_yaxis()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
