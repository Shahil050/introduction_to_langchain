# # pip install fastapi uvicorn
# from fastapi import FastAPI

# app = FastAPI()

# # Simple endpoint
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to GenAI API!"}

# # Endpoint with parameters
# @app.get("/generate/{prompt}")
# def generate_text(prompt: str, max_tokens: int = 100):
#     # In real app: call LLM here
#     return {
#         "prompt": prompt,
#         "max_tokens": max_tokens,
#         "response": f"Generated text for: {prompt}"
#     }

# # Run: uvicorn filename:app --reload










# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()

# # Define data structure
# class PromptRequest(BaseModel):
#     prompt: str
#     temperature: float = 0.7
#     max_tokens: int = 150

# # POST endpoint
# @app.post("/chat")
# def chat_completion(request: PromptRequest):
#     # Process the prompt
#     response = {
#         "prompt": request.prompt,
#         "settings": {
#             "temperature": request.temperature,
#             "max_tokens": request.max_tokens
#         },
#         "completion": "This is where LLM response goes"
#     }
#     return response






import numpy as np
import pandas as pd
from fastapi import FastAPI
from pathlib import Path

app = FastAPI()
promptpath=Path.cwd().parent/"resource/prompt_templates.csv"

# Load prompt templates
def load_templates():
    df = pd.read_csv(promptpath)
    return df

# Calculate similarity (NumPy)
def find_best_template(query_embedding, template_embeddings):
    similarities = np.array([
        np.dot(query_embedding, t) / (np.linalg.norm(query_embedding) * np.linalg.norm(t))
        for t in template_embeddings
    ])
    return np.argmax(similarities)

# API endpoint
@app.post("/find-template")
def get_template(query: str):
    # This is a simplified example showing integration
    templates = load_templates()
    return {"best_template": templates.iloc[0]['template']}
