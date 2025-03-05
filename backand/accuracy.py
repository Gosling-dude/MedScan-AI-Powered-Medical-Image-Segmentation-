import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define dataset paths and model path
TEST_DIR = "dataset/bone_fracture/test/"
MODEL_PATH = "models/bone_fractur_model.h5"

# Check if the test directory exists
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Test dataset not found! Ensure dataset/bone_fracture/test/ exists.")

# Image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16

# Data augmentation and loading for test data
datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale"
)

# Load trained model
model = load_model(MODEL_PATH)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

# Print model performance
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

