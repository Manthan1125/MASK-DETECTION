# import the necessary packages
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Set constants
INIT_LR = 1e-4
EPOCHS = 10
BS = 32
IMAGE_SIZE = (96, 96)

print("[INFO] Step 1: Loading images...")

# Paths
dataset_path = "dataset"
categories = ["with_mask", "without_mask"]

data = []
labels = []

# Loop over categories
for category in categories:
    path = os.path.join(dataset_path, category)
    print(f"[INFO] Scanning folder: {path}")
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] Unable to load image: {img_path}")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

print(f"[INFO] Total images loaded: {len(data)}")

# Encode labels
print("[INFO] Step 2: Encoding labels...")
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Convert to numpy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Train/Test Split
print("[INFO] Step 3: Splitting dataset...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Model Building
print("[INFO] Step 4: Building model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

model = Sequential([
    baseModel,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax")
])

# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Step 5: Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] Step 6: Training starting now...\n")
H = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, validation_data=(testX, testY))

# Save model
print("[INFO] Step 7: Saving model...")
model.save("mask_detector.model.h5")

print("[INFO] Done! Model saved as mask_detector.model")
