from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
MODEL_PATH = "./models/"
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess function
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = preprocess_image(file_path)
    prediction = model.predict(image)[0, :, :, 0]
    fracture_detected = np.max(prediction) > 0.5  # Threshold for fracture detection

    return jsonify({
        'filename': filename,
        'fracture_detected': bool(fracture_detected),
        'prediction_map': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
