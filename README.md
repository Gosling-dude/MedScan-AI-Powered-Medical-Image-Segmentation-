# Bone Fracture Detection API & Web App

This project provides an AI-powered medical image analysis system for bone fracture detection using X-ray images. It includes:

- A **Flask-based API** to process X-ray images and detect fractures.
- A **deep learning model** trained on labeled X-ray images.
- A **React-based frontend** where users can upload images, enter patient details, and generate medical reports.

## Features

✅ Upload X-ray images  
✅ Detect bone fractures using AI  
✅ View predictions with heatmaps  
✅ Generate downloadable medical reports  
✅ RESTful API for integration  

---

## Installation & Setup

### 1️⃣ Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Node.js 16+
- TensorFlow, OpenCV, Flask
- React.js for frontend

### 2️⃣ Clone the Repository
```sh
 git clone https://github.com/your-repo/bone-fracture-detection.git
 cd bone-fracture-detection
```

### 3️⃣ Backend Setup (Flask API)
```sh
cd backend
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4️⃣ Model Setup
Ensure your trained model file is inside the `models/` directory:
```sh
backend/models/bone_fracture_model.h5
```
If the file is missing, retrain the model or download it.

### 5️⃣ Start Backend Server
```sh
python app.py
```
This will start the API at `http://127.0.0.1:5000`

### 6️⃣ Frontend Setup (React App)
```sh
cd ../frontend
npm install
npm start
```
Access the frontend at `http://localhost:3000`

---

## API Endpoints

### 🔹 Predict Fracture (POST `/predict`)
**Request:**
```sh
POST http://127.0.0.1:5000/predict
Content-Type: multipart/form-data
```
**Form Data:**
- `file`: X-ray image (JPEG/PNG)

**Response:**
```json
{
  "filename": "xray1.png",
  "fracture_detected": true,
  "prediction_map": [[...]]
}
```

---

## Troubleshooting

### 🔴 `FileNotFoundError: models/bone_fracture_model.h5 not found`
- Ensure the model file exists at `backend/models/`
- Run `os.path.exists("models/bone_fracture_model.h5")` in Python to verify
- Use absolute path in `app.py`

### 🔴 React frontend fails to load
- Ensure backend is running (`python app.py`)
- Check API connection in `frontend/src/config.js`

---

## License
This project is open-source under the MIT License.

🚀 Happy Coding!
