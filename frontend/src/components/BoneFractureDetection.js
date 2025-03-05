import { useState } from "react";
import "../Style/boneStyle.css"
export default function BoneFractureDetection() {
  const [image, setImage] = useState(null);
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [result, setResult] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageChange = (e) => {
    const selectedImage = e.target.files[0];
    setImage(selectedImage);
    const previewUrl = URL.createObjectURL(selectedImage);
    setImagePreview(previewUrl);
  };

  const handleSubmit = async () => {
    if (!image || !name || !age) {
      alert("Please fill in all fields");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);
    formData.append("name", name);
    formData.append("age", age);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("submit", data);
      setResult(data);
    } catch (error) {
      console.error("Error uploading image", error);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h2 className="title">Bone Fracture Detection</h2>
        <div className="form">
          <div className="input-group">
            <label htmlFor="name">Patient Name</label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input"
            />
          </div>
          <div className="input-group">
            <label htmlFor="age">Age</label>
            <input
              id="age"
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="input"
            />
          </div>
          <div className="input-group">
            <label htmlFor="file">Upload X-ray Image</label>
            <input
              id="file"
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="input"
            />
          </div>
          <button onClick={handleSubmit} className="submit-btn">
            Analyze
          </button>
        </div>

        {imagePreview && (
          <div className="image-preview">
            <h3>Uploaded X-ray Image</h3>
            <img
              src={imagePreview}
              alt="Uploaded X-ray"
              className="uploaded-image"
            />
          </div>
        )}

        {result && (
          <div className="result">
            <h3>Report</h3>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Age:</strong> {age}</p>
            <p><strong>Fracture Detected:</strong> {result.fracture_detected ? "No" : "Yes"}</p>
            {result.fracture && (
              <div>
                <h4>Fracture Heatmap</h4>
                <img
                  src={`data:image/png;base64,${result.heatmap}`}
                  alt="Fracture Heatmap"
                  className="heatmap"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
