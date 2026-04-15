# 😊😢😡 Facial Emotion Recognition
### MCA (AI & ML) — Mini Project | 2nd Semester

---

## 📁 Project Structure
```
FacialEmotionRecognition/
│
├── train.py        ← Train the CNN model
├── webcam.py       ← Real-time webcam detection
├── predict.py      ← Predict from single image
├── config.py       ← All settings/constants
├── utils.py        ← Helper functions
├── archive.zip     ← FER-2013 dataset (place here)
│
├── dataset/        ← Auto-created after training
│   ├── train/
│   └── test/
│
├── models/         ← Auto-created after training
│   ├── emotion_model.keras
│   └── best_model.keras
│
└── outputs/        ← Saved plots
    ├── training_history.png
    ├── confusion_matrix.png
    └── sample_images.png
```

---

## 🚀 How to Run

### Step 1 — Setup environment
```bash
conda activate ml_env
cd ~/Desktop/FacialEmotionRecognition
```

### Step 2 — Place dataset
Place `archive.zip` (FER-2013 from Kaggle) in this folder.

### Step 3 — Train the model
```bash
python train.py
```
⏱️ Takes ~20-25 mins on M4 Mac

### Step 4 — Real-time webcam
```bash
python webcam.py
```

### Step 5 — Predict from photo
```bash
python predict.py your_photo.jpg
```

---

## 🧠 Model Architecture
- 4 Convolutional Blocks (32→64→128→256 filters)
- BatchNormalization + Dropout (prevents overfitting)
- Dense Head: 512 → 256 → 7 (Softmax)
- ~2.5M trainable parameters

## 📊 Results
- Dataset: FER-2013 (~35,000 images)
- Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Expected Accuracy: ~62–66%
