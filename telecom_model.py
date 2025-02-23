import joblib

model = joblib.load("models/telecom_model.pkl")

sample_input = [[0.3, 0.7, 0.1, 0.5, 0.9]]
prediction = model.predict(sample_input)[0]

print(f"Telecom Model Prediction: {prediction}")
