"""Train a CNN on the FER2013 dataset and save it as emotion_model.keras

1. Download fer2013.csv from Kaggle ("FER2013") and place it in this folder.
2. Run:  python train_model.py
"""
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# FER2013 label order
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

df = pd.read_csv('fer2013.csv')
X = np.array([np.array(p.split(), dtype=np.uint8).reshape(48, 48, 1) for p in df['pixels']]) / 255.0
y = to_categorical(df['emotion'], num_classes=len(EMOTIONS))

train = (df['Usage'] == 'Training').values
X_train, y_train = X[train], y[train]
X_val, y_val = X[~train], y[~train]

model = models.Sequential([
    layers.Input((48, 48, 1)),
    layers.Conv2D(32, 3, activation='relu', padding='same'), layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu', padding='same'), layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Conv2D(64, 3, activation='relu', padding='same'), layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation='relu', padding='same'), layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Conv2D(128, 3, activation='relu', padding='same'), layers.BatchNormalization(),
    layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'), layers.Dropout(0.5),
    layers.Dense(len(EMOTIONS), activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64)
model.save('emotion_model.keras')
print('Saved emotion_model.keras')
