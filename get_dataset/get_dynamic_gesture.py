import cv2
import mediapipe as mp
import numpy as np
import os
import time

GESTURE_NAME = "waving"          
SEQUENCE_LENGTH = 30           
DATASET_DIR = "dataset/Dynamic/waving"
CAMERA_INDEX = 2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

os.makedirs(os.path.join(DATASET_DIR, GESTURE_NAME), exist_ok=True)

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

    return np.array(features, dtype=np.float32)

sequence_count = len(os.listdir(os.path.join(DATASET_DIR, GESTURE_NAME)))

recording = False
frames_collected = []

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

        cv2.imshow("Dataset Recorder", image)

        key = cv2.waitKey(1) & 0xFF

        # Start recording
        if key == ord('r') and not recording:
            recording = True
            frames_collected = []
            print("Recording started...")
            time.sleep(0.3)

        # Collect frames
        if recording:
            landmarks = extract_landmarks(results)
            frames_collected.append(landmarks)

            if len(frames_collected) == SEQUENCE_LENGTH:
                sequence = np.array(frames_collected)
                filename = f"seq_{sequence_count:04d}.npy"
                path = os.path.join(DATASET_DIR, GESTURE_NAME, filename)
                np.save(path, sequence)

                print(f"Saved {path}")
                sequence_count += 1
                recording = False
                frames_collected = []
                time.sleep(0.5)

        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
