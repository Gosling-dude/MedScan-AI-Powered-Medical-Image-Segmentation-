import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define dataset paths
TRAIN_DIR = "dataset/bone_fracture/train/"
TEST_DIR = "dataset/bone_fracture/test/"
MODEL_PATH = "models/bone_fracture_model.h5"

# Ensure dataset directories exist
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Bone fracture dataset not found! Ensure dataset/bone_fracture/train/ and dataset/bone_fracture/test/ exist.")

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 10

# Data augmentation and loading
datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale"
)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale"
)

# Build CNN model for fracture classification
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: fractured or not
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = build_model()
print("🚀 Training started...")
history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# Save model
if not os.path.exists("models"):
    os.makedirs("models")
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# Plot training performance
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training Performance')
plt.show()
