# Visualizes Filter Bank CSP features with PCA.
#

# shows the 25% of test data that is used for testing 
# creates a 2D plot so the EEG feature space is easier to understand.

from functools import partial
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["toolbar"] = "None"

import matplotlib.pyplot as plt
from moabb.pipelines import FilterBank
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from training.train_filterbank import load_filterbank_data


OUTPUT_PATH = Path("reports/pca_filterbank_features.png")

def main():

    X, y, metadata = load_filterbank_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    

    feature_pipeline = make_pipeline(
        FilterBank(
            CSP(
                n_components = 6,
                reg = None,
                log = True,
                norm_trace = False,
            )
        ), SelectKBest(
            score_func = partial(mutual_info_classif, random_state = 42),
            k = 20,
        ),
    )

    X_train_features = feature_pipeline.fit_transform(X_train, y_train) # fit feature extraction pipeline

    X_test_features = feature_pipeline.transform(X_test) # transform test data

    # train lda on the selected features so we can mark waht is correctly classified in the plot

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train_features, y_train)

    y_pred = classifier.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)

# pca turns 20 selected features into 2 numbers / 2D
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_features)

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    labels = sorted(set(y_test))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 7))

    for label in labels:
        matching_points = y_test == label

        plt.scatter(
            X_test_pca[matching_points, 0],
            X_test_pca[matching_points, 1],
            label = label,
            alpha = 0.75
        )

        # mark incorrect predictions with a black x
        incorrect = y_pred != y_test

        plt.scatter(
            X_test_pca[incorrect, 0],
            X_test_pca[incorrect, 1],
            color = "black",
            marker = "x",
            s = 100,
            label = "Incorrect prediction"
        )

        plt.title("PCA Visualization of Filter Bank CSP Features")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.show()
    print("Saved PCA plot to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()