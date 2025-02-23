import joblib

# Load the trained model
model = joblib.load("models/bess_model.pkl")

# Sample prediction
sample_input = [[0.5, 0.8, 0.3, 0.2, 0.7]]
prediction = model.predict(sample_input)[0]

print(f"BESS Model Prediction: {prediction}")
