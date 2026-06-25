# trains the AI model on current data set
# creates model, trains model, tests that training
# predict.py is for new models 

#load EEG data
# split into training data and testing data
# build model
# train model
# test model
# save model


# # Path lets us create file paths that work on Mac, Windows, and Raspberry Pi.
from pathlib import Path

# joblib saves Python machine-learning models to files.
# Used to save the trained EEG classifier.
import joblib

# CSP stands for Common Spatial Patterns.
# It is a classic EEG feature-extraction method for motor imagery.
# It looks for channel patterns that help separate classes like left_hand vs feet.
from mne.decoding import CSP

# LDA stands for Linear Discriminant Analysis.
# This is the actual classifier that learns how to separate the classes.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# accuracy_score tells us what fraction of predictions were correct.
# classification_report gives more detail for each class.
from sklearn.metrics import accuracy_score, classification_report

# train_test_split splits the dataset into:
# - training data: the model is allowed to learn from this
# - testing data: hidden data used to check if the model learned for real
from sklearn.model_selection import train_test_split




# make_pipeline chains steps together.
# Our pipeline will be:

# EEG trial -> CSP feature extraction -> LDA classifier

from sklearn.pipeline import make_pipeline




# Import the function we wrote in load_data.py.
# This lets train.py reuse the loading code instead of copying it.
from load_data import load_subject


# This is where the trained model will be saved.
# The file does not need to exist yet.
MODEL_PATH = Path("models/motor_model.pkl")






def main():
    X, y, metadata = load_subject(subject_id=1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)


    # Split EEG data into train and test groups.
    #
    # test_size=0.25 means 25% of trials are kept hidden for testing.
    # random_state=42 makes the split repeatable, so you get the same split each run.
    # stratify=y keeps the class balance similar in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.25,
        random_state = 42,
        stratify = y
    )

        # Print the new shapes so we can see how much data is used for each part.
        print("Training trials:", X_train.shape, "Testing trials:", X_test.shape)


    # CSP extracts features from EEG data, and LDA classifies those features.

        model = make_pipeline(
            CSP(n_components = 8, reg = None, log = True, norm_trace = False),
            LinearDiscriminantAnalysis()
        )

    #### Train the model using eeg data and correct labels ############################
    model.fit(X_train, y_train)

    ## Ask trained model to predict labels of unseen 25% of trials (X only)
    y_pred = model.predict(X_test) # y_pred is the model's guess of the correct labels for the hidden trials


# print accuracy of model on the hidden 25% of trials
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


# print detailed report
    print("Classification report:", classification_report(y_test, y_pred))

# make sure models direcrory exists before saving
    MODEL_PATH.parent.mkdir(exist_ok=True)

# Save the trained model so we can use it later in predict.py and main.py.
    joblib.dump(model, MODEL_PATH)

# Tell me where the model was saved.
    print("Saved trained model to:", MODEL_PATH)

    if __name__ == "__main__":
    main()


