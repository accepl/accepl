from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from advanced_web_search import web_search
from bess_model import predict_bess
from fault_model import predict_fault
from finance_model import predict_finance
from grid_model import predict_grid
from loss_model import predict_loss
from maintenance_model import predict_maintenance
from military_model import predict_military
from oil_gas_model import predict_oil_gas
from renewable_model import predict_renewable
from space_model import predict_space
from telecom_model import predict_telecom

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("frontend/index.html")

# Serve logo
@app.get("/logo.jpg")
async def serve_logo():
    return FileResponse("frontend/logo.jpg")

# API for prediction
@app.post("/predict")
async def predict(request: Request, prompt: str = Form(...), model: str = Form(...)):
    model_mapping = {
        "bess": predict_bess,
        "fault": predict_fault,
        "finance": predict_finance,
        "grid": predict_grid,
        "loss": predict_loss,
        "maintenance": predict_maintenance,
        "military": predict_military,
        "oil_gas": predict_oil_gas,
        "renewable": predict_renewable,
        "space": predict_space,
        "telecom": predict_telecom,
    }
    
    if model not in model_mapping:
        return JSONResponse(content={"error": "Invalid model selected"}, status_code=400)
    
    prediction = model_mapping[model](prompt)
    return JSONResponse(content={"prediction": prediction})

# API for web search
@app.post("/web_search")
async def search(request: Request, query: str = Form(...)):
    results = web_search(query)
    return JSONResponse(content={"results": results})

@app.get("/health")
def health_check():
    return {"message": "AI Prediction & Web Search API is running!"}
