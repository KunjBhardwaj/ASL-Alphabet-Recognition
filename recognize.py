import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import urllib.request
import time

MODEL_FILE = "asl_hand_model.pkl"
HAND_MODEL = "hand_landmarker.task"

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

with open(MODEL_FILE, "rb") as f:
    data = pickle.load(f)

model = data["model"]
classes = data["classes"]

if not os.path.exists(HAND_MODEL):
    print("Downloading hand model...")
    urllib.request.urlretrieve(URL, HAND_MODEL)

base = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

last_prediction = "Nothing"
stable_prediction = "Nothing"
count = 0
last_time = time.time()
fps = 0

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(
        image,
        timestamp
    )

    prediction = "Nothing"
    confidence = 0

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        wrist = hand[0]

        points = []

        for p in hand:
            x = p.x - wrist.x
            y = p.y - wrist.y
            z = p.z - wrist.z
            points.append([x, y, z])

        points = np.array(points)

        scale = np.max(np.linalg.norm(points, axis=1))

        if scale > 0:

            points = points / scale

            features = points.flatten().reshape(1, -1)

            probabilities = model.predict_proba(features)[0]

            index = np.argmax(probabilities)

            confidence = probabilities[index]

            if confidence >= 0.30:
                prediction = str(model.classes_[index])

        h, w, _ = frame.shape

        for p in hand:
            x = int(p.x * w)
            y = int(p.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    if prediction == last_prediction:
        count += 1
    else:
        last_prediction = prediction
        count = 0

    if count >= 2:
        stable_prediction = prediction

    current_time = time.time()

    if current_time != last_time:
        fps = 1 / (current_time - last_time)

    last_time = current_time

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 150),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "ASL Recognition",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Letter: " + stable_prediction,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Confidence: " + str(round(confidence * 100, 1)) + "%",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.imshow(
        "ASL Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
detector.close()