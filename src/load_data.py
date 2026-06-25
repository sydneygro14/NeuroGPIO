# loads EEG dataset

from moaab.datasets import BNCI2014_001
from moaab.paradigms import MotorImagery


motor_imagery_classes = {"left hand", "right hand", "feet", "tongue"}

# function to load eeg data from one subject

def load_subject(subject id: int = 1):
    dataset = BNCI2014_001()
    paradigm - MotorImagery(events= motor_imagery_classes, fmin = 8, fmax = 30)

    # X = eeg data array
    # y = labels array

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    return X, y, metadata


def main():
    X, y, metadata = load_subject(subject_ID=1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First 10 labels:", y[:10])

    #print unique labels in the dataset (no repeats)
    print("Unique labels:", sorted(set(y)))

    # print first few rows of metadata
    print("Metadata:", metadata.head())

# prevents main() from running when this file is imported as a module
    if __name__ == "__main__":
    main()