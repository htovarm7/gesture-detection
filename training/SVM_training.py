import os
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib

DATASET_DIR = "dataset/Static"  
TEST_SIZE = 0.2
RANDOM_STATE = 42

X = []
y = []
label_map = {}
label_id = 0

for folder_name in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    label_map[folder_name] = label_id
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            with open(os.path.join(folder_path, file)) as f:
                row = next(csv.reader(f))
                X.append([float(x) for x in row])
                y.append(label_id)

    label_id += 1

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
labels_in_test = unique_labels(y_test)
target_names = [k for k, v in label_map.items() if v in labels_in_test]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names))

os.makedirs("weights", exist_ok=True)
joblib.dump(svm, "weights/static_pose_svm.pkl")
joblib.dump(scaler, "weights/static_pose_scaler.pkl")
joblib.dump(label_map, "weights/static_pose_labels.pkl")

print("✅ Model, scaler, and label map saved!")
