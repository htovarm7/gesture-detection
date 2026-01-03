import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf

# Load models
svm = joblib.load("./weights/static_pose_svm.pkl")
scaler = joblib.load("./weights/static_pose_scaler.pkl")
label_map = joblib.load("./weights/static_pose_labels.pkl")
inv_label_map = {v:k for k,v in label_map.items()}

lstm_model = tf.keras.models.load_model("./weights/gesture_lstm_model.keras")
gesture_labels = np.load("./weights/gesture_labels.npy")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


T = 30  #
seq_buffer = []

def extract_landmarks(results):
    # Pose
    pose = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose.extend([0]*132)

    # Left hand
    lh = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh.extend([0]*63)

    # Right hand
    rh = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh.extend([0]*63)

    return np.array(pose + lh + rh, dtype=np.float32)

cap = cv2.VideoCapture(2)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

        landmarks = extract_landmarks(results)

        landmarks_scaled = scaler.transform([landmarks])
        pose_pred_num = svm.predict(landmarks_scaled)[0]
        pose_pred_name = inv_label_map[pose_pred_num]

        seq_buffer.append(landmarks)
        if len(seq_buffer) > T:
            seq_buffer.pop(0)
        gesture_pred_name = "-"
        if len(seq_buffer) == T:
            seq_array = np.expand_dims(np.array(seq_buffer), axis=0)  
            pred_probs = lstm_model.predict(seq_array, verbose=0)
            gesture_pred_name = gesture_labels[np.argmax(pred_probs)]

        cv2.putText(image, f"Pose: {pose_pred_name}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(image, f"Gesture: {gesture_pred_name}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Pose + Gesture Detection", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()