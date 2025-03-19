import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")

try:
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Hello, how are you?")
    print(response.text)
except Exception as e:
    print("Error:", e)
