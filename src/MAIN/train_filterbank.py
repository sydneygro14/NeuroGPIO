 #Trains the best current NeuroGPIO model:
# Filter Bank CSP + LDA.

from pathlib import Path
from pickle import TRUE

from experiments.filterbank_experiment import FILTER_BANKS, MOTOR_IMAGERY_CLASSES
import joblib
from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines import FilterBank
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


MODEL_PATH = Path("models/filterbank_motor_model.pkl")

def load_filterbank_data():
    # Load the BNCI motor-imagery dataset.
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


def build_model():

    model = make_pipeline(
        FilterBank(
            CSP(
                n_components = 6,
                reg = None,
                log = True,
                norm_trace = False,
            )
        ), LinearDiscriminantAnalysis(),
    )
    return model

def main():
    X, y, metadata = load_filterbank_data()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_model()

    model.fit(X_train, y_train)
            
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(list(set(y_test)))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nlabels:", labels)
    print("confusion matrix:\n", matrix)

    MODEL_PATH.parent.mkdir(exist_ok=TRUE)
    joblib.dump(model, MODEL_PATH)

    print("Saved trained model to:", MODEL_PATH)

if __name__ == "__main__":
    main()