import streamlit as st
import google.generativeai as genai

# Configure Gemini API Key
os.getenv(api_key="GEMINI_API_KEY")

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

# Prediction using Gemini API
if st.button("Predict"):
    user_input_str = ", ".join([f"{feature}: {value}" for feature, value in zip(feature_names, user_responses)])

    # Construct prompt for Gemini
    prompt = f"""
    Given the following health data: {user_input_str}, predict the likelihood of lung cancer in percentage. 
    Explain why this percentage was given and suggest lifestyle changes or medical advice to reduce risk.
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    st.subheader("Lung Cancer Prediction Result:")
    st.write(response.text)
