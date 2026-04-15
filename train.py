
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from utils import find_dataset_paths, plot_training_history, plot_confusion_matrix

# ─── Step 1: Extract Dataset ───────────────────────────────
def extract_dataset():
    zip_locations = ['archive.zip',
                     os.path.expanduser('~/Downloads/archive.zip')]
    zip_path = next((z for z in zip_locations if os.path.exists(z)), None)

    if zip_path is None:
        print(" archive.zip not found! Please place it in the project folder.")
        exit()
    elif not os.path.exists('dataset'):
        print(f" Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('dataset')
        print(" Dataset extracted!")
    else:
        print(" Dataset already exists!")

    os.makedirs('models',  exist_ok=True)
    os.makedirs('outputs', exist_ok=True)


# ─── Step 2: Load Data ─────────────────────────────────────
def load_data(train_path, test_path):
    train_gen = ImageDataGenerator(
        rescale=1./255, rotation_range=15,
        width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=True, zoom_range=0.1
    ).flow_from_directory(
        train_path, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, color_mode='grayscale', class_mode='categorical'
    )
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, color_mode='grayscale',
        class_mode='categorical', shuffle=False
    )
    print(f"\n Train samples : {train_gen.samples}")
    print(f" Test  samples : {test_gen.samples}")
    print(f" Classes       : {train_gen.class_indices}")
    return train_gen, test_gen


# ─── Step 3: Build CNN Model ───────────────────────────────
def build_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32,(3,3), padding='same', input_shape=(IMG_SIZE,IMG_SIZE,1)),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(32,(3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64,(3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(64,(3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128,(3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128,(3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        # Block 4
        layers.Conv2D(256,(3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        # Classifier Head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256, activation='relu'), layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\n Model built | Parameters: {model.count_params():,}")
    return model


# ─── Step 4: Train ─────────────────────────────────────────
def train_model(model, train_gen, test_gen):
    print("\n🚀 Training started — please wait ~20-25 mins on M4 Mac")
    print("─" * 55)

    history = model.fit(
        train_gen, epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
    )
    model.save('models/emotion_model.keras')
    print("─" * 55)
    print("Model saved → models/emotion_model.keras")
    print("Run webcam.py for real-time detection!")
    return history


# ─── Step 5: Evaluate ──────────────────────────────────────
def evaluate_model(model, test_gen):
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\n Test Accuracy : {acc*100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
    y_true = test_gen.classes

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)


# ─── Main ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("Facial Emotion Recognition — CNN Training")
    print("  MCA (AI & ML) | 2nd Semester")
    print("=" * 55)

    extract_dataset()

    train_path, test_path = find_dataset_paths()
    if train_path is None:
        print("Could not find train/test folders in dataset!")
        exit()

    train_gen, test_gen = load_data(train_path, test_path)
    model   = build_model()
    history = train_model(model, train_gen, test_gen)
    evaluate_model(model, test_gen)
