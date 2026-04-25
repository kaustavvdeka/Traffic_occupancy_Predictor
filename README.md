# 🚗 Parking Occupancy Prediction System

A Machine Learning project that predicts **parking occupancy percentage** using image-based vehicle detection and time-based feature engineering.

---

## 📌 Overview

This project processes parking lot images, detects vehicles using computer vision techniques, and predicts occupancy using machine learning models such as **Gradient Boosting** and **Random Forest**.

The system is fully self-contained and does not require external APIs or pretrained detection models.

---

## ⚙️ Features

* 📸 Vehicle detection using OpenCV (edge + contour detection)
* 📊 Feature extraction from images (brightness, edge density, etc.)
* ⏰ Time-based feature engineering (hour, weekday, rush hours)
* 🤖 ML models:

  * Gradient Boosting Regressor
  * Random Forest Regressor
* 📈 Model evaluation (MAE, RMSE, R² score)
* 🔮 Real-time prediction from a single image
* 💾 Model saving & loading (`.pkl`)

---

## 🧠 How It Works

1. Images are processed to detect vehicles
2. Features are extracted from images
3. Time-based features are added
4. Data is scaled and fed into ML models
5. Model predicts parking occupancy (%)

Core logic implemented in:

*
*

---

## 📂 Project Structure

```
Traffic_control/
│── parkingg/
│   ├── train/
│   ├── test/
│   ├── valid/
│
│── parking_model.py
│── parking_api.py
│── eval_dashboard.py
│── parking_model.pkl
│── yolov8n.pt
│── test_results.png
│── training_results.png
│── README.md
│── .gitignore
```

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/traffic-control.git
cd traffic-control
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model

```bash
python parking_model.py
```

### Predict from an image

```python
from parking_model import ParkingPredictor

predictor = ParkingPredictor()
predictor.load_model("parking_model.pkl")

result = predictor.predict_from_image("test.jpg")
print(result)
```

---

## 📊 Example Output

```
Vehicles detected: 12
Detected occupancy: 60%
Predicted occupancy: 64%
Confidence: High
```

---

## 📈 Model Performance

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Cross-validation metrics

---

## 📦 Dependencies

* Python 3.x
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

Install via:

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib
```

---

## ⚠️ Notes

* Large files (`.pkl`, `.pt`, images) are ignored using `.gitignore`
* Update dataset path in code before running:

```python
image_folder = "path/to/your/dataset"
```

---

## 🔮 Future Improvements

* Use deep learning models (YOLO, CNN)
* Real-time camera integration
* Web dashboard for monitoring
* Deployment using Flask / FastAPI

---

## 👤 Author

Kaustav

---

## 📄 License

This project is for educational and academic use.
