import streamlit as st
import pickle
import os

# correct model path (works in Streamlit Cloud + local)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "spam_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.title("📩 Spam Email Classifier")

st.write("Enter a message and check if it's Spam or Ham")

message = st.text_area("Enter Email Text")

if st.button("Predict"):
    if message.strip():
        prediction = model.predict([message])[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter a message")
