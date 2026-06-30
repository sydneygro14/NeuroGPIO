# extracts features from EEG -- not actually needed for program, just helps me learn about the data


# NumPy is used for numerical arrays and math.
import numpy as np

# Welch's method estimates power at different frequencies.
# We use it to inspect how much power exists in a frequency band.
from scipy.signal import welch

# Import our data loader.
from data.load_data import load_subject


# Sampling rate for BCI Competition IV 2a is 250 Hz.
# That means 250 EEG samples are recorded per second.
SAMPLING_RATE = 250


def bandpower(channel_data, fmin, fmax):
    # channel_data is one EEG channel over time.
    #
    # Example shape:
    # 1001 samples
    #
    # welch() converts the time signal into frequency power estimates.
    frequencies, power = welch(channel_data, fs=SAMPLING_RATE)

    # Find which frequency bins are inside the band we care about. -- whichever bin has highest power, we 
    # know has the highest activity so we can classify
    # bandpower groups those bins into useful EEG bands
    # CSP+LDA / the classifier uses the pattern of those powers to predict the class


    band_mask = (frequencies >= fmin) & (frequencies <= fmax)

    # Average the power values inside that frequency band.
    return np.mean(power[band_mask])


def extract_trial_features(one_trial):
    # one_trial is one EEG trial.
    #
    # Shape:
    # channels x samples
    #
    # We will calculate simple average bandpower features across channels.

    # These are common EEG bands.
    mu_band = (8, 13)
    beta_band = (13, 30)

    # Store each channel's mu power here.
    mu_powers = []

    # Store each channel's beta power here.
    beta_powers = []

    # Loop through each EEG channel in this trial.
    for channel_data in one_trial:
        # Calculate mu power for this channel.
        mu_power = bandpower(channel_data, mu_band[0], mu_band[1])

        # Calculate beta power for this channel.
        beta_power = bandpower(channel_data, beta_band[0], beta_band[1])

        # Save the results.
        mu_powers.append(mu_power)
        beta_powers.append(beta_power)

    # Turn Python lists into NumPy arrays.
    mu_powers = np.array(mu_powers)
    beta_powers = np.array(beta_powers)

    # Return a small dictionary of simple features.
    return {
        "mean_mu_power": np.mean(mu_powers),
        "mean_beta_power": np.mean(beta_powers),
        "max_mu_power": np.max(mu_powers),
        "max_beta_power": np.max(beta_powers),
    }


def main():
    # Load EEG data.
    X, y, metadata = load_subject(subject_id=1)

    # Pick the first trial.
    one_trial = X[0]

    # Get the correct label for that trial.
    label = y[0]

    # Extract simple features from that trial.
    features = extract_trial_features(one_trial)

    # Print what class this trial actually belongs to.
    print("Trial label:", label)

    # Print the features we calculated.
    print("Features:")
    for feature_name, feature_value in features.items():
        print(feature_name, "=", feature_value)


if __name__ == "__main__":
    main()