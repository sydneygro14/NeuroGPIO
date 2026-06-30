# Tests feature selection after Filter Bank CSP.

# Tested Filter Bank CSP plus SelectKBest. This found the better cross-validation result: #

from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines import FilterBank

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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


def run_experiment(k_features: int):
    X, y, metadata = load_filterbank_data()

    model = make_pipeline(
        FilterBank(
            CSP(
                n_components=6,
                reg=None,
                log=True,
                norm_trace=False,
            )
        ),
        SelectKBest(
            score_func=mutual_info_classif,
            k=k_features,
        ),
        LinearDiscriminantAnalysis(),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    return scores.mean(), scores.std()


def main():
    k_options = [8, 12, 16, 20, 24, 30, 36]

    results = []

    for k_features in k_options:
        mean_accuracy, std_accuracy = run_experiment(k_features)

        row = {
            "k_features": k_features,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
        }

        results.append(row)

        print("Finished:", row)

    results = sorted(
        results,
        key=lambda row: row["mean_accuracy"],
        reverse=True,
    )

    print("\nRanked results:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()