"""Music recommendation based on facial expressions (Flask + OpenCV + Keras)."""
import json
import threading
from collections import Counter, deque
from urllib.parse import quote_plus

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

app = Flask(__name__)
model = load_model('emotion_model.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('songs.json', encoding='utf-8') as f:
    SONGS = json.load(f)

recent = deque(maxlen=10)          # smooth predictions over the last 10 frames
state = {'emotion': 'Neutral'}
lock = threading.Lock()


def frames():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError('Could not open the webcam.')
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
                face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)) / 255.0
                pred = model.predict(face.reshape(1, 48, 48, 1), verbose=0)
                recent.append(EMOTIONS[int(np.argmax(pred))])
                with lock:
                    state['emotion'] = Counter(recent).most_common(1)[0][0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (216, 75, 47), 2)
                cv2.putText(frame, state['emotion'], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (216, 75, 47), 2)
            _, buf = cv2.imencode('.jpg', frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
    finally:
        cam.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion')
def emotion():
    with lock:
        mood = state['emotion']
    songs = [{'title': s, 'url': 'https://www.youtube.com/results?search_query=' + quote_plus(s)}
             for s in SONGS.get(mood, [])]
    return jsonify({'emotion': mood, 'songs': songs})


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
