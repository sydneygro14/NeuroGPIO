# trains the AI model on current data set

from pathlib import Path

import joblib

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline



from archive.baseline.load_data import load_subject


MODEL_PATH = Path("models/motor_model.pkl")


def main():
    # Load EEG trials, labels, and metadata.
    X, y, metadata = load_subject(subject_id=1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Split data into training and testing groups.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Training trials:", X_train.shape)
    print("Testing trials:", X_test.shape)

    # CSP extracts EEG features.
    # LDA classifies those features.
    model = make_pipeline(
        CSP(n_components=12, reg=None, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )

    # Train the model on the training data.
    model.fit(X_train, y_train)

    # Predict labels for the hidden testing data.
    y_pred = model.predict(X_test)

    # Compare predictions against the real hidden labels.
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Make sure the models folder exists.
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # Save the trained model.
    joblib.dump(model, MODEL_PATH)

    print("Saved trained model to:", MODEL_PATH)


if __name__ == "__main__":
    main()