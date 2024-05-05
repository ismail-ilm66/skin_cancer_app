import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os

# Define the classes of skin cancer
classes = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma',
           'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
           'squamous cell carcinoma', 'vascular lesion']

# Load pre-trained models
model_paths = {
    "Model 1(CNN)": "cnn.h5",  # Replace 'model1.h5' with the path to your first trained model
    "Model 2(Inception Resnet V2)": "inception.h5"   # Replace 'model2.h5' with the path to your second trained model
}
models = {name: load_model(path) for name, path in model_paths.items()}

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(180, 180))  # Adjust target_size based on your model's input shape
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict(image, selected_model):
    processed_image = preprocess_image(image)
    prediction = selected_model.predict(processed_image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]
    return predicted_class, confidence

# Streamlit app
st.title('Skin Cancer Detection')

# Dropdown to select the model
selected_model = st.selectbox("Select Model", list(models.keys()))

uploaded_file = st.file_uploader("Choose an image...", type=['jpg' , 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save the uploaded file to a temporary location
    file_path = 'temp.jpg'
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    class_name, confidence = predict(file_path, models[selected_model])

    # Remove the temporary file
    os.remove(file_path)

    st.write(f"Predicted Class: {class_name}")
    st.write(f"Confidence: {confidence}")
