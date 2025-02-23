import joblib

model = joblib.load("models/military_model.pkl")

sample_input = [[0.6, 0.2, 0.9, 0.5, 0.3]]
prediction = model.predict(sample_input)[0]

print(f"Military Model Prediction: {prediction}")
