 import os
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from advanced_web_search import web_search

# ✅ Load Models (Now Directly Inside app.py)
try:
    bess_model = joblib.load("bess_model.pkl")
    fault_model = joblib.load("fault_model.pkl")
    finance_model = joblib.load("finance_model.pkl")
    grid_model = joblib.load("grid_model.pkl")
    oil_gas_model = joblib.load("oil_gas_model.pkl")
    renewable_model = joblib.load("renewable_model.pkl")
    space_model = joblib.load("space_model.pkl")
    telecom_model = joblib.load("telecom_model.pkl")
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Serve Static Frontend (HTML + Logo)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("frontend/index.html")

# ✅ Define Input Model
class InputData(BaseModel):
    input_features: list

# ✅ AI Prediction Endpoints
@app.post("/predict/bess")
async def predict_bess(data: InputData):
    try:
        result = bess_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fault")
async def predict_fault(data: InputData):
    try:
        result = fault_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/finance")
async def predict_finance(data: InputData):
    try:
        result = finance_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/grid")
async def predict_grid(data: InputData):
    try:
        result = grid_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/oil_gas")
async def predict_oil_gas(data: InputData):
    try:
        result = oil_gas_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/renewable")
async def predict_renewable(data: InputData):
    try:
        result = renewable_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/space")
async def predict_space(data: InputData):
    try:
        result = space_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/telecom")
async def predict_telecom(data: InputData):
    try:
        result = telecom_model.predict([data.input_features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ AI-Powered Web Search
@app.get("/search")
async def search_web(query: str):
    return {"results": web_search(query)}

# ✅ Run App
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
