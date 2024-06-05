import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import base64

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('./bg5.png')

# Function to preprocess and predict the uploaded image
def predict_uploaded_image(uploaded_image, model):
    # Preprocess the image
    img = np.array(uploaded_image)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0  # Normalize the image
    
    # Predict the class of the image
    pred = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(pred)
    
    return predicted_class

# Load the trained model
model_path = 'model.h5'
model = load_model(model_path)

# Streamlit web app
def main():
    st.title("Diabetic Retinopathy Detection")
    st.header("Upload an Image of a Retinal Scan")
    
    # Allow the user to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict the class when the user clicks the button
        if st.button("Predict"):
            # Get the predicted class
            predicted_class = predict_uploaded_image(image, model)
            
            # Display the prediction
            classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            st.success(f"Predicted Class: {classes[predicted_class]}")

if __name__ == "__main__":
    main()
