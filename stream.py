import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import Image
import os

# Load the pre-trained model
model_path = "/content/drive/MyDrive/mini/mask_detector.model.keras"
if not os.path.exists(model_path):
    st.error("Model file not found! Please check the path.")
else:
    mask_detector = load_model(model_path)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_mask_and_label(image, mask_detector):
    original_image = np.array(image)
    image_array = original_image.copy()
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "No face detected", 0.0

    for (x, y, w, h) in faces:
        face = image_array[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_resized = img_to_array(face_resized)
        face_resized = preprocess_input(face_resized)
        face_resized = np.expand_dims(face_resized, axis=0)

        predictions = mask_detector.predict(face_resized)
        mask, without_mask = predictions[0]
        label = "Mask" if mask > without_mask else "No Mask"
        confidence = max(mask, without_mask)

        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
        cv2.rectangle(image_array, (x, y), (x+w, y+h), color, 2)
        label_text = f"{label}: {confidence * 100:.2f}%"
        cv2.putText(image_array, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    annotated_image = Image.fromarray(image_array)
    return annotated_image, label, confidence

# Streamlit UI
st.title("Face Mask Detection")  # Title displayed only once

# Ensure a unique key for the file uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="unique_file_uploader")

if uploaded_file:
    # Process the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image (Original)", use_column_width=True)

    # Prediction button with a unique key
    if st.button("Predict", key="unique_predict_button"):
        labeled_image, label, confidence = predict_mask_and_label(image, mask_detector)
        st.image(labeled_image, caption=f"Predicted Label: {label}", use_column_width=True)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Percentage:** {confidence * 100:.2f}%")










# Step 1: Install necessary packages
!pip install pyngrok streamlit --quiet

# Step 2: Import required libraries
from pyngrok import ngrok

# Step 3: Add your ngrok authtoken
authtoken = "2qMsd2M08Ty6oSQEuhe5iCF9xAg_4UXzdHuiQpVKb4fZVq7JT"  # Your token
!ngrok config add-authtoken {authtoken}

# Step 4: Run your existing app.py
!streamlit run /content/drive/MyDrive/app.py &>/dev/null &  # Update the path if app.py is elsewhere

# Step 5: Ensure correct ngrok tunnel configuration
# This step explicitly sets up the tunnel with the correct port
tunnel = ngrok.connect(8501)  # Streamlit's default port
print(f"Your app is live at: {tunnel}")
