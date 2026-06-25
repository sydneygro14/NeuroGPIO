# sees how model predicts new datasets


 Path helps us point to files/folders in a clean way.
from pathlib import Path

# joblib lets us load the trained model that train.py saved.
import joblib

# We use the same train/test split logic as train.py.
# This lets us recreate the hidden test set.
from sklearn.model_selection import train_test_split

# Import our own data-loading function from load_data.py.
from load_data import load_subject


# This is the file where train.py saved the trained model.
MODEL_PATH = Path("models/motor_model.pkl")

TEST_TRIAL_INDEX = 0

def load_saved_model():
    model = joblib.load(MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return model


    def get_one_unseen_trial(subject_id: int = 1, trial_index: int = 0):
        X, y, metadata = load_subject(subject_id = subject_id)

        _, X_test, _, y_test = train_test_split( 
            X,
            y,
            test_size = 0.35,
            stratify = y,
            random_state = 42
        )


    one_trial = X_test[trial_index]
    correct_label = y_test[trial_index]

    return one_trial, correct_label


    def main():
    #Load the model that was trained and saved by train.py.
    model = load_saved_model()

    # Get one trial that the model has never seen before.
    trial, correct_label = get_one_unseen_trial(
        subject_id = 1,
        trial_index = TEST_TRIAL_INDEX
    )
# print shape of this one trial
    print("One trial shape:", one_trial.shape)
    one_trial_for_model = one_trial[None, :, :] # one trial in this batch since model needs more than one trial to predict

    prediction_label = prediction[0] 

    
    # Print what the model guessed.
    print("Predicted label:", predicted_label)

    # Print the real answer from the dataset.
    print("Correct label:", correct_label)

    # Print whether the model got this one trial right.
    print("Correct prediction:", predicted_label == correct_label)


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()

