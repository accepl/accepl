import joblib

model = joblib.load("models/fault_model.pkl")

sample_input = [[0.1, 0.6, 0.9, 0.3, 0.2]]
prediction = model.predict(sample_input)[0]

print(f"Fault Model Prediction: {prediction}")
