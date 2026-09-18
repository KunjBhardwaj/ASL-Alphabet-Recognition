# Real-Time ASL Alphabet Recognition System

## Project Overview

The **Real-Time ASL Alphabet Recognition System** is a computer vision and machine learning project that recognizes American Sign Language (ASL) alphabet hand gestures through a webcam.

The system detects the user's hand using MediaPipe, extracts 21 hand landmarks, converts them into normalized numerical features, and uses a trained Random Forest classifier to predict the ASL alphabet class in real time.

The project focuses on recognizing individual ASL alphabet gestures. It does not claim to perform unrestricted sentence-level sign language translation.

## Problem Statement

Communication can be difficult between people who use sign language and people who do not understand sign language. This project develops a simple computer vision system that can recognize ASL alphabet hand gestures from a live webcam feed.

The system converts a visible hand gesture into a predicted ASL alphabet class using hand landmarks and machine learning.

## Objectives

- Detect a hand from a live webcam.
- Extract 21 hand landmarks using MediaPipe.
- Convert landmarks into numerical features.
- Normalize the extracted features.
- Classify ASL alphabet gestures using a machine learning model.
- Display the predicted letter in real time.
- Display the prediction confidence.
- Maintain stable predictions across multiple frames.
- Evaluate the trained model using standard classification metrics.

## Functional Modules

### 1. Hand Detection

The webcam provides the input video. MediaPipe Hand Landmarker detects the hand and provides 21 landmark points.

**Input:** Webcam frame

**Output:** Hand landmarks

### 2. Feature Extraction

Each detected hand contains 21 landmarks. Every landmark contains X, Y and Z coordinates.

```text
21 landmarks × 3 coordinates = 63 features
```

The landmarks are converted into a 63-value feature vector.

### 3. Feature Normalization

The landmark coordinates are made relative to the wrist and scaled according to the hand size.

This reduces the effect of the hand being at different positions or distances from the camera.

### 4. ASL Classification

The normalized features are passed to the trained Random Forest classifier.

**Input:** 63 numerical features

**Output:** ASL class and prediction probability

### 5. Real-Time Recognition

The trained model is loaded and applied to each webcam frame. Multiple consecutive frames are used to stabilize the displayed prediction.

### 6. Result Visualization

The application displays:

- ASL prediction
- Confidence percentage
- Hand landmarks
- Live camera feed
- FPS
- User controls

## Non-Functional Requirements

### Performance

The system should provide real-time webcam processing with a usable frame rate.

### Usability

The application should have a simple interface that allows the user to perform a gesture in front of the webcam and see the prediction.

### Reliability

The system should handle frames where no hand is detected and should reduce unstable predictions using frame-based stabilization.

### Error Handling

The program checks for required model files, the MediaPipe hand model and webcam availability.

### Maintainability

Training and recognition are kept as separate parts so that an already trained model can be reused without retraining.

### Resource Efficiency

The recognition stage works with hand landmark features instead of repeatedly processing the complete image with a large image classification network.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam and image processing |
| MediaPipe | Hand detection and landmark extraction |
| NumPy | Numerical processing |
| Scikit-learn | Random Forest classifier |
| Pickle | Model storage and loading |
| Git | Version control |
| GitHub | Repository and submission |

## Dataset

The project uses an ASL hand gesture image dataset organized into separate classes for ASL alphabet gestures.

During training, images are processed using MediaPipe. The detected hand landmarks are converted into normalized numerical features before being used to train the classifier.

The original images are not required during real-time recognition after the trained landmark-based model has been created.

## Machine Learning Model

The project uses a **Random Forest Classifier**.

### Model Selection Rationale

Random Forest is suitable for this project because:

- The input consists of numerical landmark features.
- It can work effectively with relatively small feature vectors.
- It provides class probability estimates.
- Prediction is fast enough for real-time use.
- It is simpler to deploy than a large image-based deep learning model for this landmark-based approach.

## Input and Output

### Input

A live webcam stream containing a hand performing an ASL alphabet gesture.

### Processing

1. Capture webcam frame.
2. Convert the frame to RGB.
3. Detect the hand using MediaPipe.
4. Extract 21 landmarks.
5. Convert landmarks to 63 features.
6. Normalize the features.
7. Pass the features to the Random Forest model.
8. Obtain the predicted class and probability.
9. Apply a confidence threshold.
10. Stabilize the prediction across frames.
11. Display the result.

### Output

The application displays the predicted ASL alphabet class, confidence percentage, hand landmarks and FPS.

## System Architecture

```text
+----------------------+
|      Webcam          |
+----------+-----------+
           |
           v
+----------------------+
| OpenCV Video Input   |
+----------+-----------+
           |
           v
+----------------------+
| MediaPipe Hand       |
| Detection            |
+----------+-----------+
           |
           v
+----------------------+
| 21 Hand Landmarks    |
| X, Y, Z Coordinates  |
+----------+-----------+
           |
           v
+----------------------+
| Feature Extraction   |
| and Normalization    |
+----------+-----------+
           |
           v
+----------------------+
| Random Forest Model  |
+----------+-----------+
           |
           v
+----------------------+
| ASL Letter +         |
| Confidence           |
+----------------------+
```

## Installation

Install Python and then install the required packages:

```bash
python -m pip install --upgrade mediapipe opencv-python numpy scikit-learn
```

## Running the Project

### Step 1: Train the Model

If you want to train the model again using the dataset:

```bash
python train_model.py
```

The training process creates the trained model file:

```text
asl_hand_model.pkl
```

### Step 2: Start Recognition

Run:

```bash
python recognize.py
```

The webcam window will open.

Perform an ASL alphabet gesture in front of the camera.

### Controls

| Key | Action |
|---|---|
| Q | Quit |
| R | Reset prediction |

## Evaluation Methodology

The model should be evaluated using a separate test portion of the dataset.

The main evaluation metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Accuracy is calculated as:

```text
Correct Predictions
------------------- × 100
Total Predictions
```

The confusion matrix can be used to identify ASL classes that are frequently confused with each other.

**Important:** The actual accuracy and other metric values should be taken directly from the training/evaluation output. They should not be manually estimated.

## Testing

### Functional Testing

The following should be tested:

- Application starts correctly.
- Webcam opens correctly.
- Hand is detected.
- Hand landmarks are displayed.
- Model loads correctly.
- ASL prediction is displayed.
- Confidence is displayed.
- Q closes the application.
- R resets the prediction.

### Recognition Testing

Test the system with different:

- Hand positions
- Distances from the webcam
- Lighting conditions
- Backgrounds
- Hand orientations

### Error Testing

Test the system when:

- No hand is visible.
- The webcam is unavailable.
- The trained model is missing.
- The MediaPipe model is missing.
- Prediction confidence is low.

## Limitations

- The system focuses on individual ASL alphabet gestures.
- It does not provide unrestricted sentence-level sign language translation.
- Recognition depends on the quality and diversity of the training dataset.
- Poor lighting, occlusion or partial hand visibility can reduce recognition performance.
- The current implementation uses one detected hand.
- Dynamic gestures such as ASL J and Z may require movement information across multiple frames.
- Classifier probability should not be interpreted as a guaranteed real-world accuracy measure.

## Future Enhancements

- Add dynamic ASL gesture recognition.
- Add two-hand recognition.
- Add word and sentence formation.
- Add text-to-speech.
- Improve the graphical interface.
- Increase dataset diversity.
- Add automatic dataset collection.
- Improve performance under different lighting conditions.
- Deploy the system as a web or mobile application.

## Challenges

- Working with different versions of the MediaPipe API.
- Extracting reliable hand landmarks.
- Normalizing landmarks for different hand positions and sizes.
- Reducing unstable frame-by-frame predictions.
- Handling low-confidence predictions.
- Maintaining real-time performance.

## Learning Outcomes

This project demonstrates:

- Computer vision
- Hand landmark detection
- Feature extraction
- Feature normalization
- Machine learning classification
- Random Forest
- Real-time video processing
- Model evaluation
- Confusion matrix analysis
- Git and GitHub project organization

## References

- MediaPipe documentation
- OpenCV documentation
- Scikit-learn documentation
- ASL hand gesture dataset used for this project
