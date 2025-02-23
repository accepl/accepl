from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import joblib
import uvicorn
import requests
from bs4 import BeautifulSoup
import os

# ✅ Load AI Models
bess_model = joblib.load("bess_model.pkl")
fault_model = joblib.load("fault_model.pkl")
finance_model = joblib.load("finance_model.pkl")
grid_model = joblib.load("grid_model.pkl")
oil_gas_model = joblib.load("oil_gas_model.pkl")
renewable_model = joblib.load("renewable_model.pkl")
space_model = joblib.load("space_model.pkl")
telecom_model = joblib.load("telecom_model.pkl")

# ✅ Store models in dictionary for easy access
models = {
    "bess": bess_model,
    "fault": fault_model,
    "finance": finance_model,
    "grid": grid_model,
    "oil_gas": oil_gas_model,
    "renewable": renewable_model,
    "space": space_model,
    "telecom": telecom_model,
}

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Basic Web Scraper for Search Feature
def web_search(query):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    
    for g in soup.find_all('div', class_='tF2Cxc'):
        title = g.find('h3').text if g.find('h3') else "No Title"
        link = g.find('a')['href'] if g.find('a') else "No Link"
        results.append(f"<a href='{link}' target='_blank'>{title}</a>")
    
    return "<br>".join(results[:5]) if results else "No results found."

# ✅ Homepage with Embedded HTML
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return f"""
    <html>
    <head>
        <title>AI Prediction & Web Search</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 20px; }}
            .container {{ max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }}
            img {{ width: 150px; margin-bottom: 10px; }}
            select, input, button {{ width: 100%; padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="https://raw.githubusercontent.com/accepl/accepl/main/logo.jpg" alt="Company Logo">
            <h2>AI Prediction & Web Search</h2>
            
            <!-- AI Prediction Form -->
            <form action="/predict" method="post">
                <label>Select AI Model:</label>
                <select name="model_name">
                    <option value="bess">BESS (Battery)</option>
                    <option value="fault">Fault Detection</option>
                    <option value="finance">Financial</option>
                    <option value="grid">Grid Load</option>
                    <option value="oil_gas">Oil & Gas</option>
                    <option value="renewable">Renewable Energy</option>
                    <option value="space">Space & Aerospace</option>
                    <option value="telecom">Telecom</option>
                </select>
                <label>Enter Data (comma-separated):</label>
                <input type="text" name="user_input" placeholder="e.g. 500,85,0.07,320">
                <button type="submit">Predict</button>
            </form>
            
            <!-- Web Search Form -->
            <form action="/search" method="post">
                <label>Search the Web:</label>
                <input type="text" name="query" placeholder="Enter search query">
                <button type="submit">Search</button>
            </form>
        </div>
    </body>
    </html>
    """

# ✅ AI Prediction Endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, user_input: str = Form(...), model_name: str = Form(...)):
    if model_name in models:
        model = models[model_name]
        try:
            prediction = model.predict([[float(x) for x in user_input.split(",")]])[0]
            result = f"<h3>Prediction for {model_name}: {prediction}</h3>"
        except Exception as e:
            result = f"<h3>Error: {str(e)}</h3>"
    else:
        result = "<h3>Invalid Model Selected</h3>"

    return homepage() + f"<div class='container'>{result}</div>"

# ✅ Web Search Endpoint
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = web_search(query)
    return homepage() + f"<div class='container'><h3>Search Results:</h3>{results}</div>"

# ✅ Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
