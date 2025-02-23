import joblib

model = joblib.load("models/finance_model.pkl")

sample_input = [[1.2, 3.4, 5.6, 7.8, 9.0]]
prediction = model.predict(sample_input)[0]

print(f"Finance Model Prediction: {prediction}")
