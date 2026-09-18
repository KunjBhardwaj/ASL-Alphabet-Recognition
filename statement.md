# Project Statement

## Project Title

**Real-Time ASL Alphabet Recognition System**

## Problem Statement

Communication can be difficult between people who use sign language and people who do not understand sign language.

This project aims to develop a computer vision based system that recognizes American Sign Language (ASL) alphabet hand gestures through a webcam and displays the corresponding predicted alphabet class in real time.

## Scope

The project focuses on recognition of individual ASL alphabet hand gestures from a live webcam.

The system uses MediaPipe to detect hand landmarks and a machine learning classifier to recognize the gesture.

The project does not attempt unrestricted sentence-level sign language translation.

## Target Users

The system can be used by:

- Students learning about computer vision and machine learning.
- Students learning ASL alphabet gestures.
- Researchers or developers experimenting with hand gesture recognition.
- Users who want to demonstrate real-time gesture classification.

## High-Level Features

1. Real-time webcam input.
2. Hand detection using MediaPipe.
3. Extraction of 21 hand landmarks.
4. Conversion to 63 numerical features.
5. Landmark normalization.
6. ASL alphabet classification using Random Forest.
7. Confidence-based prediction.
8. Multi-frame prediction stabilization.
9. Real-time prediction display.
10. FPS and hand landmark visualization.
11. Model loading without retraining.
12. Basic error handling for missing model and camera.

## Input

Live webcam frames containing a hand performing an ASL alphabet gesture.

## Output

The system displays the predicted ASL alphabet class and its confidence percentage.

## Main Workflow

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
63 Normalized Features
   ↓
Random Forest Classifier
   ↓
ASL Alphabet Prediction
   ↓
Display Result
```

## Project Goal

The goal is to demonstrate how computer vision, feature extraction and machine learning can be combined to create a real-time ASL alphabet recognition application.
