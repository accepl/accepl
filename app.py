from fastapi import FastAPI
from pydantic import BaseModel
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

# User Input Model for Predictions
class PredictionInput(BaseModel):
    features: list  # List of numbers (features for model prediction)

@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: PredictionInput):
    if model_name not in models:
        return {"error": "Invalid model name or model not found"}
    
    model = models[model_name]
    prediction = model.predict([input_data.features])[0]
    return {
        "Model": model_name,
        "Input Features": input_data.features,
        "Prediction": prediction
    }

@app.get("/search/")
def search_web(query: str):
    """API endpoint for real-time web search."""
    search_results = web_search(query)

    formatted_results = []
    for result in search_results["results"]:
        formatted_results.append({
            "URL": result["url"],
            "Summary": result["summary"],
            "Keywords": ", ".join(result["keywords"])
        })

    return {
        "Query": query,
        "Total Results": len(formatted_results),
        "Results": formatted_results
    }
