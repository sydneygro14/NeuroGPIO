# Benchmark evaluation across all 9 subjects of BNCI2014_001 (BCI Competition IV 2a).
#
# For each subject, two scores are reported:
#   cross_session: train on session 1 (training day), test on session 2 (evaluation day).
#                  This is the standard competition protocol and the most honest number.
#   within_session_cv: 5-fold shuffled CV on all of that subject's trials
#                  (the same method as the subject 1 results in the README).
#
# Cohen's kappa is also reported because it was the official competition metric
# (0 = chance, 1 = perfect; for 4 balanced classes kappa = (acc - 0.25) / 0.75).
#
# Note: the pipeline settings (6 bands, 6 CSP components, k=20, 0.6-3.1 s window)
# were tuned on subject 1, so subjects 2-9 are a fair check of how well they transfer.
#
# Run from the repo root:
#   PYTHONPATH=src python src/experiments/evaluate_all_subjects.py

import csv
from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_001
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from training.train_filterbank import build_filterbank_model, get_paradigm


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "reports" / "all_subjects_results.csv"

SUBJECTS = list(range(1, 10))


def evaluate_subject(dataset, paradigm, subject_id: int):
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    # Sessions are named "0train" / "1test" in recent MOABB versions,
    # so sorting puts the training day first.
    sessions = sorted(metadata["session"].unique())
    train_mask = (metadata["session"] == sessions[0]).to_numpy()
    test_mask = (metadata["session"] == sessions[1]).to_numpy()

    # Cross-session: train on day 1, test on day 2.
    model = build_filterbank_model()
    model.fit(X[train_mask], y[train_mask])
    y_pred = model.predict(X[test_mask])

    cross_session_acc = accuracy_score(y[test_mask], y_pred)
    cross_session_kappa = cohen_kappa_score(y[test_mask], y_pred)

    # Within-session style: shuffled 5-fold CV over all trials.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_filterbank_model(), X, y, cv=cv, scoring="accuracy")

    return {
        "subject": subject_id,
        "n_train_trials": int(train_mask.sum()),
        "n_test_trials": int(test_mask.sum()),
        "cross_session_acc": cross_session_acc,
        "cross_session_kappa": cross_session_kappa,
        "within_session_cv_acc": cv_scores.mean(),
    }


def main():
    dataset = BNCI2014_001()
    paradigm = get_paradigm()

    results = []

    for subject_id in SUBJECTS:
        row = evaluate_subject(dataset, paradigm, subject_id)
        results.append(row)

        print(
            f"Subject {subject_id}: "
            f"cross-session acc = {row['cross_session_acc']:.4f}, "
            f"kappa = {row['cross_session_kappa']:.4f}, "
            f"5-fold CV acc = {row['within_session_cv_acc']:.4f}"
        )

    cross_acc = np.array([row["cross_session_acc"] for row in results])
    cross_kappa = np.array([row["cross_session_kappa"] for row in results])
    cv_acc = np.array([row["within_session_cv_acc"] for row in results])

    print(f"\nSummary across {len(results)} subjects (chance = 25%):")
    print(
        f"Cross-session accuracy: mean {cross_acc.mean():.4f} +/- {cross_acc.std():.4f} "
        f"(range {cross_acc.min():.4f} - {cross_acc.max():.4f})"
    )
    print(f"Cross-session kappa: mean {cross_kappa.mean():.4f}")
    print(
        f"5-fold CV accuracy: mean {cv_acc.mean():.4f} +/- {cv_acc.std():.4f} "
        f"(range {cv_acc.min():.4f} - {cv_acc.max():.4f})"
    )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved per-subject results to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
