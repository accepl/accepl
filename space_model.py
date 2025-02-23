import joblib

model = joblib.load("models/space_model.pkl")

sample_input = [[0.2, 0.9, 0.5, 0.6, 0.8]]
prediction = model.predict(sample_input)[0]

print(f"Space Model Prediction: {prediction}")
