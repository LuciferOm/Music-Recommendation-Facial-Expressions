# Music Recommendation Based on Facial Expressions

An AI-based system that uses a live webcam feed to detect the user's facial expression in real time and recommends music that matches the detected mood.

**Tech:** Python, OpenCV, TensorFlow/Keras, Flask

## How it works
1. OpenCV captures webcam frames and finds faces with a Haar cascade.
2. A CNN trained on FER2013 classifies the expression (7 emotions).
3. Predictions are smoothed over the last 10 frames to avoid flicker.
4. Flask shows the video feed and a song list for the detected mood (links open a YouTube search).

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Train the model
1. Download `fer2013.csv` from Kaggle (search "FER2013") into this folder.
2. Run `python train_model.py`. This creates `emotion_model.keras`.

## Run
```bash
python app.py
```
Open http://127.0.0.1:5000 and allow webcam access (it uses your local camera).

## Customize
Edit `songs.json` to change the playlist for each mood.
