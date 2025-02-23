from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from googlesearch import search
import uvicorn

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Load AI Models
try:
    bess_model = joblib.load("bess_model.pkl")
    fault_model = joblib.load("fault_model.pkl")
    finance_model = joblib.load("finance_model.pkl")
    grid_model = joblib.load("grid_model.pkl")
    oil_gas_model = joblib.load("oil_gas_model.pkl")
    renewable_model = joblib.load("renewable_model.pkl")
    space_model = joblib.load("space_model.pkl")
    telecom_model = joblib.load("telecom_model.pkl")
except:
    bess_model = fault_model = finance_model = grid_model = None
    oil_gas_model = renewable_model = space_model = telecom_model = None

# ✅ Web Search (No API Key Required)
def google_search(query, num_results=5):
    results = [url for url in search(query, num_results=num_results)]
    return results

# ✅ Frontend UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI + Web Search</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        h1 { color: #007BFF; }
        form { margin-top: 20px; }
        input, button { padding: 10px; margin: 5px; width: 300px; }
        .results { margin-top: 20px; text-align: left; }
    </style>
</head>
<body>
    <img src="https://your-logo-url.com/logo.jpg" alt="Logo" width="200">
    <h1>AI Models + Web Search</h1>
    
    <form method="post" action="/search">
        <input type="text" name="query" placeholder="Enter search query" required>
        <button type="submit">Search</button>
    </form>
    
    <h2>Predict AI Models</h2>
    
    <form method="post" action="/predict_bess">
        <h3>BESS Model</h3>
        <input type="number" name="grid_load" placeholder="Grid Load (MW)" required>
        <input type="number" name="battery_soc" placeholder="Battery SOC (%)" required>
        <input type="number" name="energy_price" placeholder="Energy Price ($/kWh)" required>
        <button type="submit">Predict BESS Efficiency</button>
    </form>

    <form method="post" action="/predict_fault">
        <h3>Fault Detection Model</h3>
        <input type="number" name="voltage" placeholder="Voltage (kV)" required>
        <input type="number" name="frequency" placeholder="Frequency (Hz)" required>
        <input type="number" name="load" placeholder="Transformer Load (MW)" required>
        <button type="submit">Predict Fault Risk</button>
    </form>

    <form method="post" action="/predict_finance">
        <h3>Financial Model</h3>
        <input type="number" name="investment" placeholder="Investment (₹ Cr)" required>
        <input type="number" name="roi" placeholder="ROI (%)" required>
        <input type="number" name="interest" placeholder="Interest Rate (%)" required>
        <button type="submit">Predict Financial Stability</button>
    </form>

    <div class="results">
        <h2>Results:</h2>
        <ul>{results}</ul>
    </div>
</body>
</html>
"""

# ✅ Home Route (Frontend)
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE.format(results="")

# ✅ Web Search Route
@app.post("/search", response_class=HTMLResponse)
async def perform_search(query: str = Form(...)):
    search_results = google_search(query)
    results_html = "".join(f"<li><a href='{url}' target='_blank'>{url}</a></li>" for url in search_results)
    return HTML_TEMPLATE.format(results=results_html)

# ✅ AI Model Prediction Routes

@app.post("/predict_bess", response_class=HTMLResponse)
async def predict_bess(grid_load: float = Form(...), battery_soc: float = Form(...), energy_price: float = Form(...)):
    if bess_model:
        df = pd.DataFrame([[grid_load, battery_soc, energy_price]], columns=['Grid Load (MW)', 'Battery SOC (%)', 'Energy Price ($/kWh)'])
        prediction = bess_model.predict(df)[0]
        return HTML_TEMPLATE.format(results=f"<li>Predicted Charging Efficiency: {prediction:.2f}%</li>")
    return HTML_TEMPLATE.format(results="<li>BESS Model Not Loaded</li>")

@app.post("/predict_fault", response_class=HTMLResponse)
async def predict_fault(voltage: float = Form(...), frequency: float = Form(...), load: float = Form(...)):
    if fault_model:
        df = pd.DataFrame([[voltage, frequency, load]], columns=['Voltage (kV)', 'Frequency (Hz)', 'Transformer Load (MW)'])
        prediction = fault_model.predict(df)[0]
        return HTML_TEMPLATE.format(results=f"<li>Fault Risk: {'Yes' if prediction == 1 else 'No'}</li>")
    return HTML_TEMPLATE.format(results="<li>Fault Model Not Loaded</li>")

@app.post("/predict_finance", response_class=HTMLResponse)
async def predict_finance(investment: float = Form(...), roi: float = Form(...), interest: float = Form(...)):
    if finance_model:
        df = pd.DataFrame([[investment, roi, interest]], columns=['Project Investment (₹ Cr)', 'ROI (%)', 'Interest Rate (%)'])
        prediction = finance_model.predict(df)[0]
        return HTML_TEMPLATE.format(results=f"<li>Financial Stability Score: {prediction:.2f}</li>")
    return HTML_TEMPLATE.format(results="<li>Finance Model Not Loaded</li>")

# ✅ Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

