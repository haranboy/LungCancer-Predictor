import streamlit as st
import google.generativeai as genai
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

# Function to load the model
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

# Function to configure the Gemini API
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

# Define the feature order
feature_order = [
    'AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
    'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL',
    'IMMUNE_WEAKNESS', 'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION',
    'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'CHEST_TIGHTNESS',
    'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE'
]

# Custom CSS for Better Styling
st.markdown("""
    <style>
        /* Custom header */
        .main-title {
            font-size: 36px;
            color: #ffffff;
            text-align: center;
            background: linear-gradient(to right, #06beb6, #48b1bf);
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
        }
        /* Prediction Box */
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        .low-risk {background: #28a745;}
        .moderate-risk {background: #ffc107; color: #333;}
        .high-risk {background: #dc3545;}
        /* Advice Box */
        .advice-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# Page Layout: Title (Top), Inputs (Left), Results (Right)
st.markdown('<div class="main-title">Lung Cancer Prediction AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])  # Left: Inputs (2x width), Right: Results (1x width)

with col1:
    st.subheader("Enter Your Health Details")

    response_dict = {feature: 0 for feature in feature_order}  # Initialize defaults

    # Numeric Inputs
    response_dict["AGE"] = st.slider("Age:", 18, 100, 40)
    response_dict["OXYGEN_SATURATION"] = st.slider("Oxygen Saturation (%):", 70, 100, 98)
    response_dict["ENERGY_LEVEL"] = st.slider("Energy Level (1-10):", 1, 10, 5)

    # Gender Selection
    response_dict["GENDER"] = 1 if st.radio("Gender:", ('Male', 'Female'), horizontal=True) == 'Female' else 0

    # Yes/No Features
    for feature in feature_order[2:]:
        response_dict[feature] = 1 if st.radio(feature.replace("_", " ").title(), ('No', 'Yes'), horizontal=True) == 'Yes' else 0

with col2:
    if st.button("Predict Chance", use_container_width=True):
        with st.spinner("Analyzing risk factors..."):
            try:
                user_input_df = pd.DataFrame([[response_dict[feature] for feature in feature_order]], columns=feature_order)
                prediction_prob = model.predict_proba(user_input_df)[0][1] * 100

                # Prediction Result Styling
                risk_class = "low-risk" if prediction_prob < 30 else "moderate-risk" if prediction_prob < 70 else "high-risk"
                st.markdown(f'<div class="prediction-box {risk_class}">Risk Level: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.progress(prediction_prob / 100)

                # AI Health Advice
                if API_KEY:
                    try:
                        prompt = f"""
                        The patient's predicted lung cancer probability is {prediction_prob:.2f}%. 
                        Provide a **detailed medical analysis** on the potential causes based on their input factors. 
                        Include **precautionary lifestyle changes, dietary adjustments, medical tests**, and when to seek medical help.
                        Structure the response as:
                        1️⃣ **Potential Causes**
                        2️⃣ **Recommended Lifestyle Changes**
                        3️⃣ **Dietary Adjustments**
                        4️⃣ **Medical Tests & Next Steps**
                        Keep the response **medically sound, structured, and detailed.**
                        """
                        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                        response = model_gemini.generate_content(prompt)
                        
                        st.subheader("🩺 AI-Generated Health Advice")
                        with st.expander("Click to view AI-generated health recommendations"):
                            if response and hasattr(response, "text"):
                                st.markdown(f'<div class="advice-box">{response.text}</div>', unsafe_allow_html=True)
                            elif response and hasattr(response, "candidates"):
                                st.markdown(f'<div class="advice-box">{response.candidates[0].content.parts[0].text}</div>', unsafe_allow_html=True)
                            else:
                                st.warning("AI did not generate a response.")
                    
                    except Exception as e:
                        st.warning(f"Gemini API Error: {e}")

                # Feature Importance Graph
                st.subheader("🔍 Factors Affecting Your Risk")
                if hasattr(model, "feature_importances_"):
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_order,
                        'Importance': model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
                    ax.set_xlabel("Importance Score")
                    ax.set_title("Feature Importance")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                else:
                    st.warning("Feature importance visualization unavailable.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
