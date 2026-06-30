# tests model accuracy

# Tests the saved model on the full hidden test set.

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data.load_data import load_subject


MODEL_PATH = Path("models/motor_model.pkl")


def main():
    # Load the trained model created by train.py.
    model = joblib.load(MODEL_PATH)

    # Load EEG data for subject 1.
    X, y, metadata = load_subject(subject_id=1)

    # Recreate the same hidden test set from train.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Predict all hidden test trials.
    y_pred = model.predict(X_test)

    # Calculate accuracy.
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Print per-class performance.
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Show which classes are confused with which.
    labels = sorted(set(y_test))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nLabels:", labels)
    print("\nConfusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
    ## confusion matrix:

#                       predicted
#               feet  left  right  tongue
# true feet       10     2      1       0
# true left        1    12      0       0
# true right       2     0     11       0
# true tongue      0     1      2      10

## so for row 1, 10 accurate feet were predicted, 1 feet was predicted as left, 2 feet were predicted as right, and 0 feet
# were predicted as tongue.

if __name__ == "__main__":
    main()