# ============================================================
#   predict.py — Predict Emotion from a Single Image
#   Usage: python predict.py your_photo.jpg
# ============================================================

import sys
import os
from tensorflow.keras.models import load_model
from utils import predict_from_image

# ─── Load Model ────────────────────────────────────────────
MODEL_PATH = 'models/emotion_model.keras'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'models/best_model.keras'

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Run train.py first.")
    exit()

model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# ─── Get Image Path ────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python predict.py your_photo.jpg")
    exit()

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    exit()

# ─── Predict ───────────────────────────────────────────────
print(f"\n🔍 Analyzing: {image_path}")
predict_from_image(image_path, model)
