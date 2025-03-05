import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
dataset_path = "backend/dataset/"
model_path = "backend/models/medical_model.h5"

# Kaggle Dataset IDs
dataset_kaggle_ids = {
    "chest_xray": "jtiptj/chest-xray-pneumoniacovid19tuberculosis",
    "brain_tumor": "dschettler8845/brats-2021-task1",
    "bone_fracture": "pkdarabi/bone-fracture-detection-computer-vision-project"
}

def download_kaggle_dataset(dataset_name, kaggle_id, dest_folder):
    """Downloads and extracts datasets from Kaggle."""
    os.makedirs(dest_folder, exist_ok=True)
    command = f'kaggle datasets download -d {kaggle_id} -p {dest_folder} --unzip'
    
    if os.system(command) == 0:
        print(f"✅ {dataset_name} downloaded and extracted.")
    else:
        print(f"❌ Error downloading {dataset_name}. Check Kaggle authentication.")

def organize_data():
    """Splits dataset into training and testing sets."""
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(images) == 0:
                print(f"❌ No images found in {category}. Check dataset files.")
                continue

            train, test = train_test_split(images, test_size=0.2, random_state=42)

            for folder in ['train', 'test']:
                os.makedirs(os.path.join(category_path, folder), exist_ok=True)

            for img in train:
                shutil.move(os.path.join(category_path, img), os.path.join(category_path, 'train', img))
            for img in test:
                shutil.move(os.path.join(category_path, img), os.path.join(category_path, 'test', img))

def train_model():
    """Trains a CNN model and saves it to backend/models/medical_model.h5."""
    img_size = (150, 150)
    batch_size = 32

    # Define dataset paths
    train_dir = os.path.join(dataset_path, "chest_xray/train")  # Change this to the correct dataset folder
    test_dir = os.path.join(dataset_path, "chest_xray/test")  # Change this to the correct dataset folder

    # Image Data Generator for augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )

    # Build CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, validation_data=test_generator, epochs=10)

    # Save the model
    model.save(model_path)
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs("backend/models/", exist_ok=True)

    # Download datasets
    for name, kaggle_id in dataset_kaggle_ids.items():
        download_kaggle_dataset(name, kaggle_id, os.path.join(dataset_path, name))

    # Organize data
    organize_data()

    # Train and save the model
    train_model()
