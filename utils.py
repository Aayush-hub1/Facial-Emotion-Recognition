import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import EMOTIONS, EMOTION_EMOJIS


def find_dataset_paths():
    candidates = [
        ('dataset/train', 'dataset/test'),
        ('dataset/train/train', 'dataset/test/test'),
    ]
    if os.path.exists('dataset'):
        for d in os.listdir('dataset'):
            t = f'dataset/{d}/train'
            if os.path.exists(t):
                candidates.append((t, f'dataset/{d}/test'))
    for tr, te in candidates:
        if os.path.exists(tr) and len(os.listdir(tr)) > 0:
            print(f'✅ Train path : {tr}')
            print(f'✅ Test  path : {te}')
            return tr, te
    return None, None


def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, (m, t) in enumerate([('accuracy','Accuracy'), ('loss','Loss')]):
        ax[i].plot(history.history[m],          label='Train', color='#2196F3', lw=2)
        ax[i].plot(history.history[f'val_{m}'], label='Val',   color='#FF5722', lw=2)
        ax[i].set_title(t, fontsize=13, fontweight='bold')
        ax[i].legend(); ax[i].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_history.png', dpi=150)
    plt.show()
    print('✅ Saved outputs/training_history.png')


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f'{e}\n{em}' for e,em in zip(EMOTIONS,EMOTION_EMOJIS)],
        yticklabels=[f'{e} {em}' for e,em in zip(EMOTIONS,EMOTION_EMOJIS)])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()
    print('✅ Saved outputs/confusion_matrix.png')


def predict_from_image(image_path, model):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print('❌ No face detected!')
        return
    for (x,y,w,h) in faces:
        face  = cv2.resize(gray[y:y+h, x:x+w], (48,48))
        inp   = np.expand_dims(face.astype('float32')/255.0, axis=[0,-1])
        preds = model.predict(inp, verbose=0)[0]
        idx   = np.argmax(preds)
        print(f'\n🎭 Emotion    : {EMOTION_EMOJIS[idx]} {EMOTIONS[idx].upper()}')
        print(f'📊 Confidence : {preds[idx]*100:.1f}%\n')
        for i,(e,p) in enumerate(zip(EMOTIONS,preds)):
            bar = '█' * int(p*30)
            print(f'  {EMOTION_EMOJIS[i]} {e:10s} {bar} {p*100:.1f}%')