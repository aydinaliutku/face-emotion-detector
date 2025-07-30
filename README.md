# 🎭 Face & Emotion Detector

Real-time face and emotion detection using Python, OpenCV, TensorFlow, and MediaPipe.

This is a beginner-friendly AI project that uses your webcam to detect faces and classify emotions using a CNN model trained on the FER2013 dataset.

---

## ✅ Features

- Real-time face detection from webcam (MediaPipe)
- Emotion classification using pre-trained Keras model
- Clean and modular codebase

---

## 🛠 Requirements

```
pip install -r requirements.txt
```

> Make sure your Python version is between **3.7 – 3.10** for TensorFlow compatibility.

---

## 📁 Folder Structure

```
face-emotion-detector/
│
├── detector.py
├── requirements.txt
├── README.md
├── .gitignore
└── model/
    └── emotion_detection_model.h5   ← Not uploaded to GitHub
```

---

## 🔗 Download the Model

The model file `emotion_detection_model.h5` is **not included** in this repo due to size limits.

You can download it from:
- [Download model.h5 from ibhanu/emotion-detection](https://github.com/ibhanu/emotion-detection/raw/master/model.h5)
- or [Hugging Face](https://huggingface.co)

Place the file inside the `model/` folder before running the code.

---

## ▶️ Run the Project

```
python detector.py
```

Press **Q** to quit the webcam stream.

---

## To-Do

- [ ] Integrate smile detection  
- [ ] Add logging and save results  
- [ ] Upload demo to Huggingface Spaces  

---

## 👤 Author

Ali Utku Aydın  
[GitHub](https://github.com/aydinaliutku) | [LinkedIn](https://www.linkedin.com/in/ali-utku-ayd%C4%B1n-643629191/)

---

> Feel free to fork this repo, improve it, and show your own touch!
