# Cross-validates the Filter Bank CSP pipeline.
#
# This gives a more honest estimate than one train/test split.

from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines import FilterBank

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def main():
    X, y, metadata = load_filterbank_data()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    model = make_pipeline(
        FilterBank(
            CSP(
                n_components=6,
                reg=None,
                log=True,
                norm_trace=False,
            )
        ),
        LinearDiscriminantAnalysis(),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    print("Fold accuracies:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())


if __name__ == "__main__":
    main()