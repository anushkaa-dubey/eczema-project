import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load trained model
model = joblib.load('eczema_model.pkl')

st.title("Eczema Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = image.resize((64, 64)).convert('RGB')
    img_array = np.array(img).flatten().reshape(1, -1)
    
    # Prediction
    prediction = model.predict(img_array)
    prob = model.predict_proba(img_array)
    
    # Output
    if prediction[0] == 1:
        st.error(f"Prediction: Eczema Detected 😟 ({prob[0][1]*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: Normal Skin 😊 ({prob[0][0]*100:.2f}% confidence)")
