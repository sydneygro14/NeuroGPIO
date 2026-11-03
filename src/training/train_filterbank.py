# Trains the best current NeuroGPIO model and saves it for runtime/main.py:
# Filter Bank CSP + SelectKBest + LDA.
#
# Steps:
# 1. Load subject 1 filter-bank EEG data.
# 2. Report 5-fold cross-validation accuracy (all scaling/CSP/selection is fit inside each fold).
# 3. Split 75% train / 25% hidden test, the same split runtime/main.py replays.
# 4. Fit on the 75% train split only, score the hidden 25%, and save the model.
#
# Run from the repo root:
#   PYTHONPATH=src python src/training/train_filterbank.py

from functools import partial
from pathlib import Path

import joblib

from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines import FilterBank

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "filterbank_motor_model.pkl"

MOTOR_IMAGERY_CLASSES = ["left_hand", "right_hand", "feet", "tongue"]

FILTER_BANKS = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 30),
]


def get_paradigm():
    return FilterBankMotorImagery(
        filters=FILTER_BANKS,
        events=MOTOR_IMAGERY_CLASSES,
        n_classes=4,
        tmin=0.6,
        tmax=3.1,
    )


def load_filterbank_data(subject_id: int = 1):
    # X shape: trials x channels x samples x filter bands
    X, y, metadata = get_paradigm().get_data(
        dataset=BNCI2014_001(),
        subjects=[subject_id],
    )

    return X, y, metadata


def build_filterbank_model():
    # Filter Bank CSP -> keep the 20 most informative features -> LDA
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


def main():
    X, y, metadata = load_filterbank_data(subject_id=1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 5-fold cross-validation (within subject 1, sessions mixed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_filterbank_model(), X, y, cv=cv, scoring="accuracy")

    print("\n5-fold CV accuracies:", cv_scores)
    print("5-fold CV mean accuracy:", cv_scores.mean())
    print("5-fold CV standard deviation:", cv_scores.std())

    # Same split runtime/main.py uses, so the replayed trials are never seen in training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_filterbank_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nHeld-out test accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Saved trained model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
