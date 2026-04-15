# ============================================================
#   webcam.py — Real-Time Emotion Detection via Webcam
#   Run: python webcam.py
#   Quit: Press Q
# ============================================================

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from config import EMOTIONS, EMOTION_COLORS

# ─── Load Model ────────────────────────────────────────────
MODEL_PATH = 'models/emotion_model.keras'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'models/best_model.keras'

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found!")
    print("   → Run train.py first to train the model.")
    exit()

print(f"✅ Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ─── Load Face Detector ────────────────────────────────────
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── Start Webcam ──────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("📷 Webcam started! Press Q to quit.")
print("─" * 35)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame!")
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))

    # No face message
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    for (x,y,w,h) in faces:
        # Preprocess
        face  = cv2.resize(gray[y:y+h, x:x+w], (48,48))
        inp   = np.expand_dims(face.astype('float32')/255.0, axis=[0,-1])
        preds = model.predict(inp, verbose=0)[0]
        idx   = np.argmax(preds)
        emotion    = EMOTIONS[idx]
        confidence = preds[idx] * 100
        color      = EMOTION_COLORS[emotion]

        # Bounding box
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        # Label
        label = f"{emotion.upper()}  {confidence:.0f}%"
        cv2.rectangle(frame, (x,y-40), (x+w,y), (0,0,0), -1)
        cv2.putText(frame, label, (x+5,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Probability bars (top-left corner)
        for i,(e,p) in enumerate(zip(EMOTIONS, preds)):
            bar_w = int(p * 120)
            bar_y = 20 + i * 22
            cv2.rectangle(frame, (10,bar_y), (10+bar_w, bar_y+16), EMOTION_COLORS[e], -1)
            cv2.putText(frame, f"{e[:3].upper()} {p*100:.0f}%",
                        (135, bar_y+13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.imshow("Facial Emotion Recognition | Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed. Goodbye!")
