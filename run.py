import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from google.colab.patches import cv2_imshow

# Load the face detector model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the mask detector model
mask_detector = load_model("/content/drive/MyDrive/mini/mask_detector.model.keras")

# Load the input image
image_path = "/content/mm.jpeg"
image = cv2.imread(image_path)
orig = image.copy()

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Loop over the detected faces
for (x, y, w, h) in faces:
    # Extract the face ROI
    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # Predict mask or no mask
    (mask, without_mask) = mask_detector.predict(face)[0]
    print(f"Mask Probability: {mask:.4f}, No Mask Probability: {without_mask:.4f}")

    # Determine the class label and color
    if mask > without_mask:
        label = "Mask"
        color = (0, 255, 0)  # Green for Mask
        confidence = mask * 100  # Confidence percentage for Mask
    else:
        label = "No Mask"
        color = (0, 0, 255)  # Red for No Mask
        confidence = without_mask * 100  # Confidence percentage for No Mask

    # Display the label and bounding box rectangle on the output frame
    cv2.putText(image, f"{label}: {confidence:.2f}%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

# Display the result in Colab
cv2_imshow(image)

# Save the result (if needed)
output_path = "/content/output_image.jpg"
# cv2.imwrite(output_path, image)
