import joblib

model = joblib.load("models/loss_model.pkl")

sample_input = [[0.7, 0.5, 0.2, 0.3, 0.9]]
prediction = model.predict(sample_input)[0]

print(f"Loss Model Prediction: {prediction}")
