import joblib

model = joblib.load("models/maintenance_model.pkl")

sample_input = [[0.4, 0.8, 0.7, 0.1, 0.5]]
prediction = model.predict(sample_input)[0]

print(f"Maintenance Model Prediction: {prediction}")
