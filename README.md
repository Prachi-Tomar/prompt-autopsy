# Prompt Autopsy
A forensic tool to compare LLM responses to the same prompt across models and explain why they differ. MVP includes:
- FastAPI endpoint /compare for multi-model calls
- Embedding-based similarity
- Token diff and placeholders for logprobs
- Streamlit UI for side-by-side viewing

## Experiments
The Experiments feature allows you to run a grid of prompts across multiple models, temperatures, system prompts, and seeds.
It computes per-model drift and stability metrics to help you understand how different parameters affect model outputs.
- Drift: How much a model's responses vary across different parameter settings
- Stability: How consistent a model's responses are (cosine similarity to centroid)

## Quick start
1) python -m venv .venv && source .venv/bin/activate (or .venv\\Scripts\\activate on Windows)
2) pip install -r requirements.txt
3) cp .env.example .env and add your API keys
4) uvicorn backend.app:app --reload
5) streamlit run frontend/streamlit_app.py

## Notes
- Logprobs are placeholders and may depend on model support
- Embeddings computed with sentence-transformers
- Use at your own risk. MIT licensed.