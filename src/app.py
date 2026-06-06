import streamlit as st
import joblib
import os

# Path works locally + Streamlit Cloud
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "spam_model.pkl")

st.title("📩 Spam Email Classifier")

st.write("Enter a message and check if it's Spam or Ham")

# Load model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

message = st.text_area("Enter Email Text")

if st.button("Predict"):
    if message.strip():
        prediction = model.predict([message])[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter a message")
