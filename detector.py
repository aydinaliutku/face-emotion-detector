#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 1) Initialize MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# 2) Load the trained Keras emotion model
model = load_model("model/emotion_detection_model.h5")

# 3) Emotion classes (FER2013 order)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# 4) Open the camera
cap = cv2.VideoCapture(0)

# For performance, analyze every N frames
SKIP_FRAMES = 5
frame_counter = 0
cached_faces = []      # list of (x,y,w,h)
cached_emotions = []   # list of labels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_counter += 1

    # Perform face detection and emotion analysis every SKIP_FRAMES frames
    if frame_counter % SKIP_FRAMES == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        cached_faces = []
        cached_emotions = []

        if results.detections:
            for detection in results.detections:
                # Convert relative bbox to pixel coords
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)

                # Expand ROI for full-face coverage
                margin_w = int(0.1 * w)
                margin_h = int(0.1 * h)
                x1 = max(0, x - margin_w)
                y1 = max(0, y - margin_h // 2)
                x2 = min(frame_width, x + w + margin_w)
                y2 = min(frame_height, y + h + margin_h)

                # Crop, preprocess, and predict
                face_img = gray[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (48, 48))
                face_img = face_img.astype("float32") / 255.0
                face_img = np.expand_dims(face_img, axis=-1)      # (48,48,1)
                face_img = np.expand_dims(face_img, axis=0)       # (1,48,48,1)

                preds = model.predict(face_img, verbose=0)[0]
                idx = np.argmax(preds)
                confidence = preds[idx]
                label = f"{emotion_labels[idx]} {confidence*100:0.0f}%"

                cached_faces.append((x1, y1, x2 - x1, y2 - y1))
                cached_emotions.append(label)

    # Draw cached results on the frame
    for idx, (x, y, w, h) in enumerate(cached_faces):
        label = cached_emotions[idx] if idx < len(cached_emotions) else "Not found"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

