import joblib

model = joblib.load("models/grid_model.pkl")

sample_input = [[0.3, 0.6, 0.1, 0.4, 0.8]]
prediction = model.predict(sample_input)[0]

print(f"Grid Model Prediction: {prediction}")
