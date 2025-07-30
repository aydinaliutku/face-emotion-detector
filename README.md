\# Face \& Emotion Detector 



Real-time face and (mock) emotion detection using Python, OpenCV, and MediaPipe.



This is a beginner-friendly AI project that uses your webcam to detect faces and randomly displays an emotion label. It serves as a foundation for further development with real emotion classification models.



\## Features



\- Real-time face detection from webcam

\- Random emotion labeling (for demo purposes)

\- Built with OpenCV, Numpy, TensorFlow and MediaPipe



\## Requirements



```bash

pip install -r requirements.txt

Make sure your Python version is 3.7 - 3.10 (for TensorFlow compatibility).


\## Folder Structure

face-emotion-detector/
│
├── detector.py
├── requirements.txt
├── README.md
├── .gitignore
└── model/
    └── emotion_detection_model.h5   ← Not uploaded to GitHub

\## Download the Model
The model file emotion_detection_model.h5 is not included in the repo due to size.
You can download it from Google Drive or Hugging Face.

Place the file inside the /model folder before running.

\## Run the Project

python detector.py

\## To-Do

-Integrate smile detection

-Add logging and save results

-Upload to Huggingface Spaces

\## Author

Ali Utku Aydın



