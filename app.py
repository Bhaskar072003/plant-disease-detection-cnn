import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the path to your model
MODEL_PATH = "https://drive.google.com/file/d/1rEh8-tW0bhoJIa7yvdjZshZZQ4KmLais/view?usp=drive_link"

# Load the trained model
@st.cache_resource  # Cache model loading for better performance
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"⚠ Error loading model: {e}")
        return None

model = load_trained_model()

# Define the class names
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

# Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf, and the model will predict its condition.")

# Upload an image
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼 Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((192, 192))  # Resize to match model input shape
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    if model:
        prediction = model.predict(img_array)[0]
        
        # Check if the model output shape matches expected classes
        if len(prediction) == len(CLASS_NAMES):
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.subheader("📊 Prediction Result")
            st.write(f"**Class:** {CLASS_NAMES[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            st.error(f"⚠ Model output mismatch: Expected {len(CLASS_NAMES)} classes, but got {prediction.shape}")
    else:
        st.error("⚠ Model is not loaded. Please check the file path.")

# Run the Streamlit app
if __name__ == "__main__":
    st._is_running_with_streamlit = True
