import joblib

model = joblib.load("models/oil_gas_model.pkl")

sample_input = [[1.0, 0.5, 0.2, 0.8, 0.6]]
prediction = model.predict(sample_input)[0]

print(f"Oil & Gas Model Prediction: {prediction}")
