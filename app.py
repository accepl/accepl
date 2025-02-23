import os
import joblib
import uvicorn
import openai
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from bs4 import BeautifulSoup
from googlesearch import search
from newspaper import Article
import yake
import re

# ✅ Ensure models are loaded
MODEL_DIR = "models"

def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    return joblib.load(model_path) if os.path.exists(model_path) else None

# ✅ Load all models
models = {
    "bess": load_model("bess_model"),
    "fault": load_model("fault_model"),
    "finance": load_model("finance_model"),
    "grid": load_model("grid_model"),
    "oil_gas": load_model("oil_gas_model"),
    "renewable": load_model("renewable_model"),
    "space": load_model("space_model"),
    "telecom": load_model("telecom_model"),
    "maintenance": load_model("maintenance_model"),
    "military": load_model("military_model"),
    "loss": load_model("loss_model")
}

# ✅ Web scraping & AI functions
openai.api_key = "your_openai_api_key"

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def google_search(query, num_results=5):
    return [url for url in search(query, num_results=num_results)]

def extract_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return clean_text(article.text[:5000])
    except Exception as e:
        return f"Error extracting content: {e}"

def extract_keywords(text, num_keywords=5):
    kw_extractor = yake.KeywordExtractor(n=1, top=num_keywords)
    return [word for word, _ in kw_extractor.extract_keywords(text)]

def summarize_content(content):
    if not content or len(content) < 100:
        return "Not enough content to summarize."
    
    prompt = f"Summarize this article and provide key insights:\n\n{content}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# ✅ FastAPI App
app = FastAPI()

# ✅ Render the web page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AI Prediction & Web Search</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; }
                h1 { color: #007bff; }
                form { margin-top: 20px; }
                input { padding: 10px; width: 300px; }
                button { padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <img src="/logo.jpg" alt="Logo" width="200">
            <h1>AI Model Prediction & Web Search</h1>
            <form action="/predict" method="post">
                <input type="text" name="model" placeholder="Enter Model Name (e.g., bess, grid)">
                <input type="text" name="data" placeholder="Enter Data (comma-separated values)">
                <button type="submit">Predict</button>
            </form>
            <form action="/websearch" method="post">
                <input type="text" name="query" placeholder="Enter Search Query">
                <button type="submit">Search</button>
            </form>
        </body>
    </html>
    """

# ✅ Prediction API
@app.post("/predict")
def predict(model: str = Form(...), data: str = Form(...)):
    if model not in models or models[model] is None:
        return {"error": f"Model '{model}' not found."}
    
    try:
        input_data = [float(x) for x in data.split(",")]
        model_instance = models[model]
        prediction = model_instance.predict([input_data])[0]
        return {"model": model, "prediction": prediction}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ✅ Web Search API
@app.post("/websearch")
def web_search(query: str = Form(...), num_results: int = 3):
    search_results = google_search(query, num_results)
    
    structured_results = []
    for url in search_results:
        content = extract_content(url)
        summary = summarize_content(content)
        keywords = extract_keywords(content)

        structured_results.append({
            "url": url,
            "summary": summary,
            "keywords": keywords
        })

    return {"query": query, "results": structured_results}

# ✅ Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
