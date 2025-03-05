from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load trained model
MODEL_PATH = "models/bone_fractur_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess function
def preprocess_image(image_path):
    # Read image in grayscale (since the model is trained on grayscale images)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))  # Resize to match the input size of the model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (224, 224, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 1)
    return image

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file to the server
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image and make a prediction
    image = preprocess_image(file_path)
    prediction = model.predict(image)
    print("Prediction:")
    print(prediction)
    # Since the model output is a single value (probability), get the prediction
    fracture_detected = prediction[0] > 0.5  # Use a threshold of 0.5 for binary classification

    print("fracture:", fracture_detected)
    return jsonify({
        'filename': filename,
        'fracture_detected': bool(fracture_detected),
        'prediction_probability': float(prediction[0])  # Return the prediction probability
    })

if __name__ == '__main__':
    app.run(debug=True)
