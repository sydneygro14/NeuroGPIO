# Tests CSP regularization and log settings. (goes through all combinations)

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from load_data import load_subject


def run_experiment(reg, log: bool):
    # Load EEG data using the best time window found so far.
    X, y, metadata = load_subject(subject_id=1, tmin=0.6, tmax=3.1)

    # Split into train/test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Build model with this experiment's CSP settings.
    model = make_pipeline(
        CSP(
            n_components=12,
            reg=reg,
            log=log,
            norm_trace=False,
        ),
        LinearDiscriminantAnalysis(),
    )

    # Train model.
    model.fit(X_train, y_train)

    # Predict hidden test set.
    y_pred = model.predict(X_test)

    # Return accuracy.
    return accuracy_score(y_test, y_pred)


def main():
    reg_options = [None, "ledoit_wolf", "oas"]
    log_options = [True, False]

    results = []

    for reg in reg_options:
        for log in log_options:
            accuracy = run_experiment(reg=reg, log=log)

            row = {
                "reg": reg,
                "log": log,
                "accuracy": accuracy,
            }

            results.append(row)

            print("Finished:", row)

    # Sort best accuracy first.
    results = sorted(results, key=lambda row: row["accuracy"], reverse=True)

    print("\nRanked results:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()