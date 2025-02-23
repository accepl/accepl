import logging
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
import joblib
import os
from advanced_web_search import web_search
from datetime import datetime

app = FastAPI()

# Set up logging
logging.basicConfig(filename="api_usage.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Define API Keys
API_KEYS = {"your-secure-api-key-123": "User1"}

# Authentication function
def authenticate(api_key: str = Header(None)):
    if api_key not in API_KEYS:
        logging.warning(f"Unauthorized access attempt with API Key: {api_key}")
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")
    
    logging.info(f"Authenticated request from user: {API_KEYS[api_key]}")
    return API_KEYS[api_key]

# Load AI models
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

models = {key: joblib.load(path) for key, path in MODEL_PATHS.items() if os.path.exists(path)}

@app.get("/")
def home():
    return {"message": "AI Prediction & Web Search API is running!"}

class PredictionInput(BaseModel):
    features: list  # List of numbers for prediction input

@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: PredictionInput, api_key: str = Depends(authenticate)):
    if model_name not in models:
        logging.warning(f"Invalid model request: {model_name}")
        raise HTTPException(status_code=400, detail="Invalid model name")

    model = models[model_name]
    prediction = model.predict([input_data.features])[0]

    logging.info(f"Prediction request: Model={model_name}, Features={input_data.features}, Result={prediction}")

    return {
        "Model": model_name,
        "Input Features": input_data.features,
        "Prediction": prediction
    }

@app.get("/search/")
def search_web(query: str, api_key: str = Depends(authenticate)):
    """API endpoint for real-time web search."""
    search_results = web_search(query)

    formatted_results = [{"URL": r["url"], "Summary": r["summary"], "Keywords": ", ".join(r["keywords"])} for r in search_results["results"]]

    logging.info(f"Search request: Query={query}, Results={len(formatted_results)}")

    return {
        "Query": query,
        "Total Results": len(formatted_results),
        "Results": formatted_results
    }
