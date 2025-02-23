from fastapi import FastAPI
import joblib
import os
from advanced_web_search import web_search

app = FastAPI()

# Define the local paths for all models
MODEL_PATHS = {
    "bess": "bess_model.pkl",
    "fault": "fault_model.pkl",
    "finance": "finance_model.pkl",
    "grid": "grid_model.pkl",
    "loss": "loss_model.pkl",
    "maintenance": "maintenance_model.pkl",
    "military": "military_model.pkl",
    "oil_gas": "oil_gas_model.pkl",
    "renewable": "renewable_model.pkl",
    "space": "space_model.pkl",
    "telecom": "telecom_model.pkl",
}

# Load all models into memory
models = {}
for key, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[key] = joblib.load(path)
    else:
        print(f"Warning: {key} model file not found at {path}")

@app.get("/")
def home():
    return {"message": "AI Prediction & Web Search API is running!"}

@app.get("/predict/{model_name}")
def predict(model_name: str):
    if model_name not in models:
        return {"error": "Invalid model name or model not found"}
    
    model = models[model_name]
    sample_input = [[0.5, 0.8, 0.3, 0.2, 0.7]]  # Replace with actual input format
    prediction = model.predict(sample_input)[0]
    return {f"{model_name}_prediction": prediction}

@app.get("/search/")
def search_web(query: str):
    """API endpoint for real-time web search."""
    search_results = web_search(query)
    return search_results
