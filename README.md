NeuroGPIO is a Raspberry Pi BCI project that classifies prerecorded motor-imagery EEG data into left hand, right hand, feet, and tongue movements. The current model uses a Filter Bank CSP + LDA pipeline and reached 87.5% accuracy on one test split, with an average of 83.86% accuracy across 5-fold cross-validation. The predicted class is sent to a simulated GPIO output now, and later it will control LEDs on a custom PCB connected to the Raspberry Pi.


| Pipeline | Best Single Split | 5-Fold CV Mean |

| CSP + LDA baseline | 84.72% | 81.43% |
The final Filter Bank CSP + feature-selection pipeline reached 87.5% on a held-out split and 85.94% mean accuracy across 5-fold cross-validation.