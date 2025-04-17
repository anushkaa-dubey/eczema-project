import streamlit as st
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants (MUST match training)
TRAIN_IMG_SIZE = (64, 64)  # Changed from 128 to match training

# Load all trained models
try:
    models = {
        "Random Forest": joblib.load('random_forest_model.pkl'),
        "SVM": joblib.load('svm_model.pkl'),
        "K-Nearest Neighbors": joblib.load('k-nearest_neighbors_model.pkl'),
        "Neural Network (MLP)": joblib.load('neural_network_(mlp)_model.pkl'),
        "CNN": load_model('cnn_model.keras')  # Changed from .h5 to .keras
    }
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Title 
st.title("Eczema Detection ")
st.markdown("""
This app uses multiple machine learning models to detect eczema in skin images.
Upload an image to see predictions from different models.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an image of skin", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    try:
        # Preprocess the image (MUST match training preprocessing)
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize(TRAIN_IMG_SIZE)  # Changed to 64x64
        
        # Prepare image for traditional models
        img_array_flat = np.array(img).flatten().reshape(1, -1)
        
        # Prepare image for CNN (normalized)
        img_array_cnn = (np.array(img) / 255.0).reshape(1, *TRAIN_IMG_SIZE, 3)
        
        # Create tabs for each model
        tabs = st.tabs([model_name for model_name in models.keys()])
        
        for tab, (model_name, model) in zip(tabs, models.items()):
            with tab:
                if model_name == "CNN":
                    # CNN prediction
                    prediction_proba = model.predict(img_array_cnn)[0][0]
                    prediction = 1 if prediction_proba > 0.5 else 0
                    eczema_prob = prediction_proba if prediction == 1 else 1 - prediction_proba
                else:
                    # Traditional model prediction
                    prediction_proba = model.predict_proba(img_array_flat)[0]
                    prediction = model.predict(img_array_flat)[0]
                    eczema_prob = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                
                #  result
                if prediction == 1:
                    st.error(f"⚠️ Eczema Detected")
                    st.metric("Confidence", f"{eczema_prob*100:.2f}%")
                else:
                    st.success(f"✅ Normal Skin")
                    st.metric("Confidence", f"{eczema_prob*100:.2f}%")
                
                # probability distribution
                prob_dist = {
                    'Normal': (1 - eczema_prob) if prediction == 1 else eczema_prob,
                    'Eczema': eczema_prob if prediction == 1 else (1 - eczema_prob)
                }
                st.bar_chart(prob_dist)
                
    except Exception as e:
        st.error(f"Error processing image: {e}")