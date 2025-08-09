# Prompt Autopsy
[![CI](https://github.com/Prachi-Tomar/prompt-autopsy/actions/workflows/ci.yml/badge.svg)](https://github.com/Prachi-Tomar/prompt-autopsy/actions/workflows/ci.yml)
A forensic tool to compare LLM responses to the same prompt across models and explain why they differ. MVP includes:
- FastAPI endpoint /compare for multi-model calls
- Embedding-based similarity
- Token diff and placeholders for logprobs
- Logprob difference analysis between models
- Automatic detection of temperature/system prompt influence on output similarity and hallucination risk
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

## Mock Mode (no API keys required)
1) In `.env`, set `MOCK_MODE=1`
2) Start backend and frontend as usual
3) Run comparisons and experiments — all charts/tables will render with synthetic data
4) Set `MOCK_MODE=0` to go back to live APIs

## Notes
- Logprobs are placeholders and may depend on model support
- Embeddings computed with sentence-transformers
- Use at your own risk. MIT licensed.

### Cost estimates
Costs are estimated from editable tables in `backend/analysis/pricing.py` (USD per 1K tokens). Update them to match your account's pricing. Estimates are informational only.