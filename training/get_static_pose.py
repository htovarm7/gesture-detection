import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

DATASET_DIR = "dataset/Static/standing"
CAMERA_INDEX = 2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

pose_dir = os.path.join(DATASET_DIR)
os.makedirs(pose_dir, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

def extract_landmarks(results):
    features = []

    # Pose landmarks
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0] * (33 * 4))

    # Left hand landmarks
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0] * (21 * 3))

    # Right hand landmarks
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0] * (21 * 3))

    return features

sample_count = len(os.listdir(pose_dir))

def mouse_callback(event, x, y, flags, param):
    global sample_count
    if event == cv2.EVENT_LBUTTONDOWN:
        landmarks = extract_landmarks(param['results'])
        filename = f"sample_{sample_count:04d}.csv"
        filepath = os.path.join(pose_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(landmarks)

        print(f"Saved {filepath}")
        sample_count += 1

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Static Pose Recorder", image)

        cv2.setMouseCallback("Static Pose Recorder", mouse_callback, {'results': results})

        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
