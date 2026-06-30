# final program that ties everything together

# 1. Load EEG trials and labels.

# 2. Split the data:
#    75% for learning
#    25% hidden for testing

# 3. Build pipeline:
#    CSP -> LDA

# 4. Train:
#    model.fit(X_train, y_train)

# 5. Predict:
#    model.predict(X_test)

# 6. Score:
#    compare y_pred to y_test

# 7. Save:
#    save trained model for later Raspberry Pi replay

# Final program that ties everything together.
#
# This file loads a trained model, replays hidden EEG trials,
# predicts the motor-imagery class, and sends that prediction
# to the output layer.

from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from train_filterbank import load_filterbank_data
from gpio_simulation import show_prediction, all_leds_off


MODEL_PATH = Path("models/filterbank_motor_model.pkl") ## filterbank model path

def main():
    # Load the trained model saved by train.py.
    model = joblib.load(MODEL_PATH)

    # Load data for subject 1.
    X, y, metadata = load_filterbank_data(subject_id=1)

    # Split data the same way as train.py.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Go through each hidden EEG test trial.
    for trial_index, one_trial in enumerate(X_test):
        # Reshape from channels x samples to 1 x channels x samples.
        one_trial_for_model = one_trial[None, :, :]

        # Make prediction for this trial.
        prediction = model.predict(one_trial_for_model)

        # prediction is an array, even though it predicted one trial.
        # [0] gets the actual predicted label.
        predicted_label = prediction[0]

        # Get the correct answer for comparison.
        correct_label = y_test[trial_index]

        print("Trial:", trial_index)
        print("Predicted:", predicted_label, "| Correct:", correct_label)

        # Send the predicted label to the simulated/real PCB output.
        show_prediction(predicted_label)

        # Wait for you to press Enter before moving to the next trial.
        input("Press Enter for next EEG trial...")

    # Turn off all LEDs when the program finishes.
    all_leds_off()


if __name__ == "__main__":
    main()

#   For each hidden EEG test trial:
# 1. reshape it so the model accepts it
# 2. predict the class
# 3. print predicted vs correct
# 4. light the matching PCB LED
# 5. wait for Enter