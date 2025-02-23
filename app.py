import os
import joblib
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from train_models import (
    train_bess_model, train_fault_model, train_finance_model, train_grid_model,
    train_oil_gas_model, train_renewable_model, train_space_model, train_telecom_model
)

app = FastAPI()

# ✅ Train models if missing
models = {
    "bess_model.pkl": train_bess_model,
    "fault_model.pkl": train_fault_model,
    "finance_model.pkl": train_finance_model,
    "grid_model.pkl": train_grid_model,
    "oil_gas_model.pkl": train_oil_gas_model,
    "renewable_model.pkl": train_renewable_model,
    "space_model.pkl": train_space_model,
    "telecom_model.pkl": train_telecom_model
}

for model_file, train_function in models.items():
    if not os.path.exists(model_file):
        print(f"⚡ Training {model_file}...")
        train_function()

# ✅ Load trained models
bess_model = joblib.load("bess_model.pkl")
fault_model = joblib.load("fault_model.pkl")
finance_model = joblib.load("finance_model.pkl")
grid_model = joblib.load("grid_model.pkl")
oil_gas_model = joblib.load("oil_gas_model.pkl")
renewable_model = joblib.load("renewable_model.pkl")
space_model = joblib.load("space_model.pkl")
telecom_model = joblib.load("telecom_model.pkl")

# ✅ Web search function
def google_search(query):
    try:
        response = requests.get(f"https://www.googleapis.com/customsearch/v1?q={query}&key=YOUR_API_KEY&cx=YOUR_CX_ID")
        results = response.json().get("items", [])
        return [item["link"] for item in results[:5]]
    except Exception as e:
        return [str(e)]

# ✅ Home page with logo
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>AI & Web Search</title></head>
        <body style='text-align: center;'>
            <img src='/static/logo.jpg' alt='Logo' width='300'><br>
            <h1>✅ AI Models Loaded & Web Search Ready</h1>
            <form action='/search'>
                <input type='text' name='q' placeholder='Enter search query'>
                <input type='submit' value='Search'>
            </form>
        </body>
    </html>
    """

# ✅ Web search endpoint
@app.get("/search")
def search(q: str = Query(..., title="Search Query")):
    return {"results": google_search(q)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
