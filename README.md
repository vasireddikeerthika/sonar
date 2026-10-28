# Sonar

Speech emotion recognition for interview practice.

- wav2vec2 embeddings + logistic regression
- Trained on RAVDESS and CREMA-D
- Speaker-independent 6-fold CV: 59.3 +/- 6.8% (8 emotions, chance 12.5%)
- Hand-crafted features baseline: 47.6 +/- 8.2%
- Streamlit app with role and DSA interviews, dashboard and history

Limitation: accuracy on a live laptop mic is lower than on studio test data, due to domain shift.