# NeuroGPIO TODO

## Current Goal

Build the software pipeline for a Raspberry Pi EEG classifier that replays prerecorded motor-imagery EEG data, trains a model, predicts body-part motor imagery classes, and sends the predicted class to the PCB LED output board.

## Build Order

1. Load prerecorded EEG data and labels.
2. Inspect `X` and `y` shapes so the dataset structure is understood.
3. Split data into training and testing sets.
4. Train the first CSP + LDA classifier.
5. Evaluate accuracy on unseen test trials.
6. Save the trained model to `models/motor_model.pkl`.
7. Load the saved model in `predict.py` and predict one unseen trial.
8. Add GPIO output mapping for the PCB LEDs.
9. Build `main.py` to replay trials and light the matching LED.
10. Add PCA visualization to project results.

## PCA Visualization Task

Use PCA as an exploration and portfolio visualization tool, not as the first main classifier.

Planned PCA views:

- Flatten EEG trials or use extracted features.
- Reduce high-dimensional EEG data to 2 principal components.
- Plot each EEG trial as a point.
- Color points by true class: `left_hand`, `right_hand`, `feet`, `tongue`.
- Later, mark correct vs incorrect model predictions on the PCA plot.

Purpose:

- Show how high-dimensional EEG trials cluster or overlap.
- Compare raw EEG structure with classifier feature-space structure.
- Explain why CSP is used for classification while PCA is useful for visualization.
