import os
import joblib

# Corrected file path
model_path = os.path.join(os.path.dirname(__file__), "bess_model.pkl")
model = joblib.load(model_path)

def predict_bess(input_data):
    return model.predict([input_data])
