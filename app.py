import streamlit as st
import requests

# Define the FastAPI endpoint
FASTAPI_ENDPOINT = "http://localhost:8000/predict"  # Use the service name defined in docker-compose.yml

def remove_empty_lines(text):
    cleaned_text = text.replace('\n', ' ').strip()
    return cleaned_text

def make_prediction(text):
    text = remove_empty_lines(text)
    response = requests.post(FASTAPI_ENDPOINT, json={"text": text})
    
    if response.status_code == 200:
        result = response.json().get("result")
        return result
    else:
        return None  # Handle the error case

# Streamlit UI
st.title("Text Classification App")

# Add your UI components and interaction logic here
text_input = st.text_area("Enter text:")
if st.button("Classify"):
    prediction = make_prediction(text_input)
    
    if prediction is not None:
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Error occurred during prediction.")