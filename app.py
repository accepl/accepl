from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import AI Models (Updated Paths)
from models.bess_model import bess_function
from models.fault_model import fault_function
from models.finance_model import finance_function
from models.grid_model import grid_function
from models.loss_model import loss_function
from models.maintenance_model import maintenance_function
from models.military_model import military_function
from models.oil_gas_model import oil_gas_function
from models.renewable_model import renewable_function
from models.space_model import space_function
from models.telecom_model import telecom_function
from advanced_web_search import web_search

app = FastAPI()

# Serve Static Files (Logo, etc.)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Root Route - HTML UI
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Prediction & Web Search</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
            img { max-width: 200px; margin-bottom: 20px; }
            input, button { padding: 10px; margin: 5px; font-size: 16px; }
        </style>
    </head>
    <body>
        <img src="/static/logo.jpg" alt="Logo">
        <h1>AI Prediction & Web Search</h1>
        <form action="/predict" method="post">
            <input type="text" name="query" placeholder="Enter your query" required>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """

# Prediction Endpoint
@app.post("/predict")
async def predict(query: str = Form(...)):
    response = web_search(query)  # Call web search function
    return {"query": query, "result": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
