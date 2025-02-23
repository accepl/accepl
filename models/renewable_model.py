import joblib

model = joblib.load("models/renewable_model.pkl")

sample_input = [[0.9, 0.7, 0.4, 0.3, 0.1]]
prediction = model.predict(sample_input)[0]

print(f"Renewable Model Prediction: {prediction}")
