import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATASET_DIR = "dataset/Dynamic" 
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
        if file.endswith(".npy"):
            seq = np.load(os.path.join(folder_path, file))  
            X.append(seq)
            y.append(folder_name)  

    label_id += 1

X = np.array(X, dtype=np.float32)  
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label map:", label_map)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_onehot
)

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

num_classes = y_onehot.shape[1]
T, F = X.shape[1], X.shape[2]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(T, F)),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

np.save("./weights/gesture_labels.npy", encoder.classes_)
model.save("./weights/gesture_lstm_model.keras")
print("✅ LSTM gesture model saved in .keras format")

