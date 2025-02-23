import requests
from bs4 import BeautifulSoup
from googlesearch import search
from newspaper import Article
import openai
import nltk
import yake
import re

# Ensure nltk resources are available
nltk.download("punkt")

# OpenAI API Key (Replace with your actual API Key)
OPENAI_API_KEY = "your_openai_api_key"

def clean_text(text):
    """Cleans extracted text by removing unnecessary spaces and newlines."""
    text = re.sub(r'\n+', '\n', text)  # Remove excessive new lines
    text = re.sub(r'\s+', ' ', text)   # Remove excessive spaces
    return text.strip()

def google_search(query, num_results=5):
    """Perform a Google search and return the top N results."""
    search_results = []
    for result in search(query, num_results=num_results):
        search_results.append(result)
    return search_results

def extract_content(url):
    """Extracts main content from an article or webpage."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return clean_text(article.text[:5000])  # Limit to 5000 characters
    except Exception as e:
        return f"Error extracting content: {e}"

def extract_keywords(text, num_keywords=5):
    """Extracts the top keywords from a given text using YAKE."""
    kw_extractor = yake.KeywordExtractor(n=1, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [word for word, _ in keywords]

def summarize_content(content):
    """Summarizes extracted content using OpenAI GPT-4."""
    if not content or len(content) < 100:
        return "Not enough content to summarize."

    prompt = f"Summarize this article and provide key insights:\n\n{content}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )

    return response["choices"][0]["message"]["content"]

def web_search(query, num_results=5):
    """Performs an advanced web search, extracts insights, and summarizes key points."""
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

# Example Usage
if __name__ == "__main__":
    query = "Latest AI advancements in Renewable Energy"
    search_output = web_search(query)

    for idx, result in enumerate(search_output["results"]):
        print(f"\n🔹 **Result {idx + 1}:** {result['url']}")
        print(f"📌 **Summary:** {result['summary']}")
        print(f"🔑 **Keywords:** {', '.join(result['keywords'])}\n")
