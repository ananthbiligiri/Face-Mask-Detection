import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import Image
import os

# Load the pre-trained model
model_path = "/content/drive/MyDrive/mini/mask_detector.model.keras"  # Update with the correct path
if not os.path.exists(model_path):
    st.error("Model file not found! Please check the path.")
else:
    mask_detector = load_model(model_path)

# Load the face detection model (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict mask on detected face
def predict_mask_and_label(image, mask_detector):
    # Convert the image to numpy array for processing
    original_image = np.array(image)  # Preserve the original image
    image_array = original_image.copy()  # Work on a copy for annotation
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for face detection

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "No face detected", 0.0  # No face detected case

    # Process each face
    for (x, y, w, h) in faces:
        face = image_array[y:y+h, x:x+w]  # Extract face region
        face_resized = cv2.resize(face, (224, 224))  # Resize the face to fit the model input
        face_resized = img_to_array(face_resized)  # Convert the face image to array
        face_resized = preprocess_input(face_resized)  # Preprocess the face image
        face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

        # Predict mask or no mask
        predictions = mask_detector.predict(face_resized)
        mask, without_mask = predictions[0]
        label = "Mask" if mask > without_mask else "No Mask"
        confidence = max(mask, without_mask)

        # Annotate the copy with a rectangular box and label
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(image_array, (x, y), (x+w, y+h), color, 2)  # Draw rectangle around face
        label_text = f"{label}: {confidence * 100:.2f}%"  # Label text with confidence
        cv2.putText(image_array, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Convert annotated image to RGB for display
    image_rgb_annotated = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image_rgb_annotated), label, confidence

# Streamlit UI
st.title("Face Mask Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image (Original)", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        labeled_image, label, confidence = predict_mask_and_label(image, mask_detector)

        # Show the original image with the annotated results
        st.image(labeled_image, caption=f"Predicted Label: {label}", use_column_width=True)

        # Display the prediction results
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
