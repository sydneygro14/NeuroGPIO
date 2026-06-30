# Tests Filter Bank CSP.
#
# Regular CSP uses one broad frequency band, like 8-30 Hz.
# Filter Bank CSP splits EEG into smaller bands, runs CSP on each band,
# then combines all those features before classification.

from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines import FilterBank

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


MOTOR_IMAGERY_CLASSES = ["left_hand", "right_hand", "feet", "tongue"]

FILTER_BANKS = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 30),
]


def load_filterbank_data():
    dataset = BNCI2014_001()

    paradigm = FilterBankMotorImagery(
        filters=FILTER_BANKS,
        events=MOTOR_IMAGERY_CLASSES,
        n_classes=4,
        tmin=0.6,
        tmax=3.1,
    )

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

    return X, y, metadata


def run_experiment(n_components: int):
    X, y, metadata = load_filterbank_data()

    print("X shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        FilterBank(
            CSP(
                n_components=n_components,
                reg=None,
                log=True,
                norm_trace=False,
            )
        ),
        LinearDiscriminantAnalysis(),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def main():
    component_options = [2, 4, 6]

    results = []

    for n_components in component_options:
        accuracy = run_experiment(n_components)

        row = {
            "filter_banks": FILTER_BANKS,
            "csp_components_per_band": n_components,
            "accuracy": accuracy,
        }

        results.append(row)

        print("Finished:", row)

    results = sorted(results, key=lambda row: row["accuracy"], reverse=True)

    print("\nRanked results:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()