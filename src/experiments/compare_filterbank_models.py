# Compares plain Filter Bank CSP vs Filter Bank CSP + feature selection.
#
# Both pipelines use the same cross-validation splits.

from functools import partial

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


def build_plain_filterbank_model():
    return make_pipeline(
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


def build_feature_selection_model():
    return make_pipeline(
        FilterBank(
            CSP(
                n_components=6,
                reg=None,
                log=True,
                norm_trace=False,
            )
        ),
        SelectKBest(
            score_func=partial(mutual_info_classif, random_state=42),
            k=20,
        ),
        LinearDiscriminantAnalysis(),
    )


def print_results(name: str, scores):
    print("\n", name)
    print("Fold accuracies:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())


def main():
    X, y, metadata = load_filterbank_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    plain_scores = cross_val_score(
        build_plain_filterbank_model(),
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    feature_selection_scores = cross_val_score(
        build_feature_selection_model(),
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    print_results("Plain Filter Bank CSP + LDA", plain_scores)
    print_results("Filter Bank CSP + SelectKBest + LDA", feature_selection_scores)

    improvement = feature_selection_scores.mean() - plain_scores.mean()

    print("\nImprovement:")
    print(improvement)


if __name__ == "__main__":
    main()