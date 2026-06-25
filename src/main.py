# final program that ties everything together

# 1. Load prerecorded EEG trials
# 2. Split into train/test data
# 3. Filter EEG, probably 8-30 Hz for motor imagery
# 4. Extract useful features or use CSP
# 5. Train classifier
# 6. Test classifier on hidden trials
# 7. Save trained model
# 8. Replay test trials
# 9. Convert prediction to GPIO LED output

# 1. Load EEG trials and labels.

# 2. Split the data:
#    75% for learning
#    25% hidden for testing

# 3. Build pipeline:
#    CSP -> LDA

# 4. Train:
#    model.fit(X_train, y_train)

# 5. Predict:
#    model.predict(X_test)

# 6. Score:
#    compare y_pred to y_test

# 7. Save:
#    save trained model for later Raspberry Pi replay