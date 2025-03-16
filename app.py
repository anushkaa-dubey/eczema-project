import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load trained model
model = joblib.load('eczema_model.pkl')

# Title
st.title("Eczema Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload an image of skin", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (Must match training!)
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((32, 32))  # Same size as training!
    img_array = np.array(img).flatten().reshape(1, -1)

    # Make prediction
    prediction_proba = model.predict_proba(img_array)[0]
    prediction = model.predict(img_array)[0]

    # Display result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Eczema Detected with {prediction_proba[1]*100:.2f}% probability")
    else:
        st.success(f"✅ Normal Skin with {prediction_proba[0]*100:.2f}% probability")
