"""
ASL HAND GESTURE MODEL TRAINER
==============================
Reads an ASL image dataset arranged in class folders, for example:

ASL_Processed_Images/
└── asl_processed/
    └── train/
        ├── 0/
        ├── 1/
        ├── ...
        ├── 9/
        ├── A/
        ├── B/
        └── ...
            └── image files

Pipeline:
Image -> MediaPipe Hand Landmarks (21 x 3) -> normalization -> Random Forest

Output:
    asl_hand_model.pkl
    hand_landmarker.task

Run:
    python train_model.py

Or:
    python train_model.py "C:\\path\\to\\asl_processed\\train"
"""

from pathlib import Path
import sys
import os
import time
import pickle
import urllib.request
from collections import Counter

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

MODEL_FILE = "hand_landmarker.task"
OUTPUT_MODEL = "asl_hand_model.pkl"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_IMAGES_PER_CLASS = 2

# These are the locations the program will automatically try.
DATASET_CANDIDATES = [
    Path("ASL_Processed_Images") / "asl_processed" / "train",
    Path("asl_processed") / "train",
    Path("train"),
]

# ---------------------------------------------------------------------
# LIBRARY CHECK
# ---------------------------------------------------------------------

def check_libraries():
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        print("\nMissing libraries:")
        for item in missing:
            print("  -", item)
        print("\nInstall them with:")
        print("python -m pip install --upgrade " + " ".join(missing))
        sys.exit(1)


check_libraries()

import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# DATASET PATH
# ---------------------------------------------------------------------

def find_dataset():
    """
    Finds the train folder automatically.

    Priority:
    1. Path supplied on command line.
    2. Common folders relative to this script.
    """

    if len(sys.argv) > 1:
        supplied = Path(sys.argv[1]).expanduser()
        if supplied.exists() and supplied.is_dir():
            return supplied.resolve()

        print(f"\nERROR: Dataset folder does not exist:\n{supplied}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir / p for p in DATASET_CANDIDATES
    ]

    # Also try the current working directory.
    candidates += [Path.cwd() / p for p in DATASET_CANDIDATES]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    print("\nCould not automatically find the dataset.")
    print("Your screenshots show the dataset should end in:")
    print("    ASL_Processed_Images\\asl_processed\\train")
    print("\nRun the program by giving the exact train folder:")
    print(r'python train_model.py "C:\path\to\ASL_Processed_Images\asl_processed\train"')
    sys.exit(1)


# ---------------------------------------------------------------------
# MEDIAPIPE MODEL
# ---------------------------------------------------------------------

def download_hand_model():
    model_path = Path(__file__).resolve().parent / MODEL_FILE

    if model_path.exists() and model_path.stat().st_size > 100000:
        return model_path

    print("\nMediaPipe hand model not found.")
    print("Downloading hand_landmarker.task ...")

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        if model_path.exists():
            try:
                model_path.unlink()
            except Exception:
                pass

        print("\nERROR: Could not download MediaPipe hand model.")
        print("Check your internet connection and run again.")
        print("Error:", exc)
        sys.exit(1)

    if not model_path.exists() or model_path.stat().st_size < 100000:
        print("\nERROR: Downloaded MediaPipe model appears invalid.")
        sys.exit(1)

    print("MediaPipe model downloaded successfully.")
    return model_path


def create_hand_detector(model_path):
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------

def extract_features(hand_landmarks):
    """
    Converts 21 MediaPipe landmarks into 63 normalized values.

    Landmark coordinates:
        x, y, z

    Normalization:
        1. Wrist becomes (0, 0, 0).
        2. All coordinates are divided by the maximum distance
           from the wrist.

    This makes the classifier less sensitive to hand position
    and hand size.
    """

    if hand_landmarks is None or len(hand_landmarks) != 21:
        return None

    wrist = hand_landmarks[0]

    points = []
    for landmark in hand_landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z
        points.append([x, y, z])

    points = np.asarray(points, dtype=np.float32)

    scale = np.max(np.linalg.norm(points, axis=1))

    if not np.isfinite(scale) or scale < 1e-7:
        return None

    points /= scale

    return points.flatten()


def extract_from_image(image, detector):
    """
    Detects one hand in an OpenCV BGR image and returns
    the 63 normalized landmark features.
    """

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    return extract_features(result.hand_landmarks[0])


# ---------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------

def get_image_files(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_dataset(dataset_dir, detector):
    """
    Expected format:

        train/
            A/
                image1.jpg
                image2.jpg
            B/
                image1.jpg
                ...
    """

    class_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )

    if not class_dirs:
        print("\nERROR: No class folders found.")
        print("Expected folders such as 0, 1, ..., 9, A, B, C, ...")
        sys.exit(1)

    print("\nClasses found:")
    print("  " + ", ".join(p.name for p in class_dirs))

    features = []
    labels = []

    total_images = 0
    total_detected = 0
    skipped_classes = []

    print("\nExtracting MediaPipe landmarks...")
    print("This can take some time depending on the dataset size.\n")

    for class_dir in class_dirs:
        label = class_dir.name
        image_files = get_image_files(class_dir)

        if len(image_files) < MIN_IMAGES_PER_CLASS:
            skipped_classes.append(label)
            print(
                f"[SKIP] {label}: only {len(image_files)} image(s)"
            )
            continue

        class_detected = 0

        print(
            f"[{label:>3}] {len(image_files):>5} images",
            end=" -> ",
            flush=True
        )

        for image_path in image_files:
            total_images += 1

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            feature_vector = extract_from_image(image, detector)

            if feature_vector is None:
                continue

            features.append(feature_vector)
            labels.append(label)
            class_detected += 1
            total_detected += 1

        print(f"{class_detected} hands detected")

    print("\n----------------------------------------")
    print(f"Images examined : {total_images}")
    print(f"Hands detected  : {total_detected}")
    print("----------------------------------------")

    if skipped_classes:
        print(
            "Skipped classes:",
            ", ".join(skipped_classes)
        )

    if len(features) < 10:
        print("\nERROR: Too few usable samples.")
        print("MediaPipe is probably not detecting the hands correctly.")
        sys.exit(1)

    class_counts = Counter(labels)

    print("\nUsable samples per class:")
    for label in sorted(class_counts):
        print(f"  {label:>3} : {class_counts[label]}")

    # A class with only one detected image cannot participate safely
    # in a stratified train/test split.
    bad_classes = [
        label for label, count in class_counts.items()
        if count < 2
    ]

    if bad_classes:
        print(
            "\nERROR: These classes have fewer than 2 detected samples:"
        )
        print(", ".join(sorted(bad_classes)))
        print(
            "\nTry using more images or check those class folders."
        )
        sys.exit(1)

    return np.asarray(features, dtype=np.float32), np.asarray(labels)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_model(X, y):
    class_names = sorted(np.unique(y).tolist())

    print("\nTraining classes:")
    print(", ".join(class_names))

    # If a class is very small, stratification can fail.
    min_class_count = min(Counter(y).values())

    if min_class_count < 5:
        print(
            "\nWARNING: Some classes have fewer than 5 usable images."
        )
        print(
            "The model can still train, but the evaluation will be less reliable."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError as exc:
        print("\nCould not create a stratified train/test split.")
        print("Error:", exc)
        sys.exit(1)

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print("\nTraining Random Forest classifier...")

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    start = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTraining completed in {training_time:.2f} seconds.")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print("=====================")

    print(
        classification_report(
            y_test,
            predictions,
            labels=class_names,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX")
    print("================")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_names
    )

    print("Labels:", class_names)
    print(matrix)

    # Save everything required by the recognition program.
    model_data = {
        "model": classifier,
        "classes": class_names,
        "feature_count": X.shape[1],
        "accuracy": float(accuracy),
        "random_state": RANDOM_STATE,
        "description": (
            "ASL hand gesture classifier using "
            "MediaPipe 21-point normalized landmarks "
            "and Random Forest."
        ),
    }

    output_path = Path(__file__).resolve().parent / OUTPUT_MODEL

    with open(output_path, "wb") as file:
        pickle.dump(model_data, file)

    print("\n========================================")
    print("MODEL SAVED SUCCESSFULLY")
    print("========================================")
    print(output_path)
    print(f"Features: {X.shape[1]}")
    print(f"Classes : {len(class_names)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("========================================")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("       ASL HAND GESTURE MODEL TRAINER")
    print("=" * 60)

    dataset_dir = find_dataset()

    print("\nDataset folder:")
    print(dataset_dir)

    model_path = download_hand_model()

    print("\nStarting MediaPipe Hand Landmarker...")
    detector = create_hand_detector(model_path)

    try:
        X, y = load_dataset(dataset_dir, detector)
    finally:
        detector.close()

    train_model(X, y)

    print("\nDone.")
    print("\nNext step:")
    print("Run:")
    print("    python recognize_speech.py")


if __name__ == "__main__":
    main()
