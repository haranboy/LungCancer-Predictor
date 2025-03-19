import google.generativeai as genai

genai.configure(api_key="AIzaSyBN6WujsTkSVMKmB52urK7hr8LY4aLzOFg")

try:
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Hello, how are you?")
    print(response.text)
except Exception as e:
    print("Error:", e)
