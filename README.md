NeuroGPIO is a Raspberry Pi BCI project that classifies prerecorded motor-imagery EEG data into left hand, right hand, feet, and tongue movements. The current model uses a Filter Bank CSP + LDA pipeline and reached 87.5% accuracy on one test split, with an average of 83.86% accuracy across 5-fold cross-validation. The predicted class is sent to a simulated GPIO output now, and later it will control LEDs on a custom PCB connected to the Raspberry Pi.


| Pipeline | Best Single Split | 5-Fold CV Mean |

| CSP + LDA baseline | 84.72% | 81.43% |
The final Filter Bank CSP + feature-selection pipeline reached 87.5% on a held-out split and 85.94% mean accuracy across 5-fold cross-validation.



###########################################################################################

Prerecorded motor-imagery EEG data is loaded from the BNCI2014_001 dataset.

Four EEG classes are used: left_hand, right_hand, feet, and tongue.

Each EEG trial is cropped to the most useful time window: 0.6-3.1 seconds.

The EEG signal is split into multiple frequency bands from 8-30 Hz.

Common Spatial Patterns, or CSP, extracts spatial EEG features from each frequency band.

Feature selection keeps the most useful CSP features and removes weaker/noisier ones.

Linear Discriminant Analysis, or LDA, classifies the selected features into one of the four motor-imagery classes.

The trained model is saved as filterbank_motor_model.pkl.

During runtime, the saved model predicts the class of each EEG trial.

The predicted class is mapped to a GPIO output.

On a laptop, GPIO output is simulated in the terminal.

On a Raspberry Pi, GPIO output will activate the matching LED on the custom PCB.

#############################################################################################


