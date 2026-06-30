# loads EEG dataset

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


# These are the four motor-imagery classes we want from the dataset.
# They match the labels used by the BNCI2014_001 dataset.
motor_imagery_classes = ["left_hand", "right_hand", "feet", "tongue"]


# This function loads EEG data from one subject.
#
# subject_id is the person number from the dataset.
# subject_id: int = 1 means:
# - the input is called subject_id
# - it should be an integer
# - if we do not give a subject_id, Python uses 1 by default
def load_subject(subject_id: int = 1, tmin: float = 0.6, tmax: float = 3.1):    # Create the dataset object.
    dataset = BNCI2014_001()

    # Create the motor-imagery loader.
    #
    # events tells MOABB which classes we want.
    # n_classes=4 tells MOABB we are using four classes.
    # fmin=8 and fmax=30 keep the EEG frequencies most useful for motor imagery.
    paradigm = MotorImagery(
        events=motor_imagery_classes,
        n_classes=4,
        fmin=8,
        fmax=30,
        tmin = tmin,
        tmax = tmax,
    )

    # Load the actual EEG data.
    
    # X = EEG data array
    # y = label array
    # metadata = extra trial information
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    return X, y, metadata

def main():
    # Load subject 1 so we can inspect the data.
    X, y, metadata = load_subject(subject_id=1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First 10 labels:", y[:10])

    # Print unique labels in the dataset, with no repeats.
    print("Unique labels:", sorted(set(y)))

    # Print the first few rows of metadata.
    print("Metadata:")
    print(metadata.head())


# This only runs main() when we run this exact file directly.
# It does not run main() when another file imports load_subject.
if __name__ == "__main__":
    main()