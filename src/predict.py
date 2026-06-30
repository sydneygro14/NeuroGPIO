# sees how model predicts new datasets

# Loads the trained EEG model and uses it to predict one unseen EEG trial.

from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from load_data import load_subject


MODEL_PATH = Path("models/motor_model.pkl")
TEST_TRIAL_INDEX = 0


def load_saved_model():
    # Load the model file created by train.py.
    model = joblib.load(MODEL_PATH)

    return model


def get_one_unseen_trial(subject_id: int = 1, trial_index: int = 0):
    # Load the same dataset.
    X, y, metadata = load_subject(subject_id=subject_id)

    # Split the data the same way train.py did.
    # Because random_state and stratify match train.py, this recreates the same hidden test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Pick one EEG trial from the hidden test set.
    one_trial = X_test[trial_index]
    correct_label = y_test[trial_index]

    return one_trial, correct_label


def main():
    # Load the saved trained model.
    model = load_saved_model()

    # Get one trial that the model has never trained on.
    trial, correct_label = get_one_unseen_trial(
        subject_id=1,
        trial_index=TEST_TRIAL_INDEX,
    )

    # The model expects a batch of trials.
    # trial starts as shape: channels x samples
    # trial[None, :, :] changes it to: 1 x channels x samples
    trial_for_model = trial[None, :, :]

    # Ask the model to predict the class.
    prediction = model.predict(trial_for_model)[0]

    print("Predicted label:", prediction)
    print("Correct label:", correct_label)


if __name__ == "__main__":
    main()