from fastapi import FastAPI
from backend.models.schemas import CompareRequest, CompareResponse, ModelResult, ExperimentRequest, ExperimentRun, DriftStats, ExperimentResponse
from backend.adapters.openai_adapter import OpenAIAdapter
from backend.adapters.anthropic_adapter import AnthropicAdapter
from backend.analysis.embeddings import embed_texts, pairwise_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import asyncio
import random
from backend.analysis.hallucination import compute_hallucination_risk
from backend.analysis.diff import html_token_diff, unified_token_diff
from backend.utils.settings import MOCK_MODE
from backend.analysis.logprob_diff import align_tokens_and_logprobs, summarize_logprob_diff
from backend.analysis.parameter_influence import compute_parameter_influence

app = FastAPI(title="Prompt Autopsy API")

# Mock response functions
def mock_compare_response(req: CompareRequest) -> CompareResponse:
    # Seed randomness for deterministic results
    random.seed(42)
    np.random.seed(42)
    
    # Define models to use
    models = ["gpt-4o", "claude-3-opus"]
    
    # Sample output texts
    output_texts = {
        "gpt-4o": "Quantum computing uses qubits that can be both 0 and 1 at once, unlike classical bits which are either 0 or 1. This property, called superposition, allows quantum computers to process information in parallel, potentially solving certain problems much faster than classical computers. Additionally, quantum computers can leverage entanglement, where qubits become correlated in such a way that the state of one qubit can depend on the state of another, even over large distances.",
        "claude-3-opus": "Think of qubits as coins spinning in the air - while spinning, they're neither fully heads nor tails but a probabilistic mixture of both. This 'superposition' lets quantum computers explore multiple solutions simultaneously. When these spinning coins become 'entangled', measuring one coin instantly affects the other, no matter the distance between them - Einstein called this 'spooky action at a distance'."
    }
    
    results = []
    for model in models:
        # Generate logprobs (50 floats around -0.5 to -2.0)
        logprobs = [random.uniform(-2.0, -0.5) for _ in range(50)]
        
        # Generate embedding (32-dim list of small floats)
        embedding = [float(x) for x in np.random.uniform(-0.5, 0.5, 32)]
        
        # Hallucination risk and reasons
        if model == "gpt-4o":
            hallucination_risk = 22.5
            reasons = ["Has references/citations (-8.0)", "Low-confidence tokens (+6.0)", "Hedging/speculative language (+4.0)"]
        else:  # claude-3-opus
            hallucination_risk = 28.7
            reasons = ["Specifics without references (+8.0)", "No token-level confidence available (+8.0)", "Hedging/speculative language (+4.0)"]
        
        results.append(ModelResult(
            model=model,
            output_text=output_texts[model],
            tokens=output_texts[model].split(),
            logprobs=logprobs,
            embedding=embedding,
            hallucination_risk=hallucination_risk,
            hallucination_reasons=reasons
        ))
    
    # Create embedding similarity matrix (2x2 with 1.0 diagonal and ~0.90 off-diagonal)
    embedding_similarity = {
        "gpt-4o": {"gpt-4o": 1.0, "claude-3-opus": 0.91},
        "claude-3-opus": {"gpt-4o": 0.91, "claude-3-opus": 1.0}
    }
    
    # Summaries with risk_highlight
    summaries = {
        "note": "This is a mock response for demonstration purposes.",
        "risk_highlight": "Highest estimated hallucination risk: claude-3-opus (28.7/100)."
    }
    
    # Token diffs
    token_diffs = {}
    html_diff = "<div>Sample HTML diff between <span class='pa-del'>quantum computing</span> and <span class='pa-ins'>quantum mechanics</span></div>"
    unified_diff = """--- gpt-4o
+++ claude-3-opus
@@ -1 +1 @@
-Quantum computing uses qubits...
+Think of qubits as coins...
"""
    token_diffs["gpt-4o||claude-3-opus"] = {"html": html_diff, "unified": unified_diff}
# Logprob differences
    logprob_diffs = {}
    # Align tokens and compute differences
    aligned = align_tokens_and_logprobs(
        results[0].tokens, results[0].logprobs,
        results[1].tokens, results[1].logprobs
    )
    summary = summarize_logprob_diff(aligned)
    if summary:
        logprob_diffs["gpt-4o||claude-3-opus"] = summary
    
    return CompareResponse(
        results=results,
        embedding_similarity=embedding_similarity,
        summaries=summaries,
        token_diffs=token_diffs,
        logprob_diffs=logprob_diffs
    )

def mock_experiment_response(req: ExperimentRequest) -> ExperimentResponse:
    # Seed randomness for deterministic results
    random.seed(42)
    np.random.seed(42)
    
    # Define models and temperatures
    models = ["gpt-4o", "claude-3-opus"]
    temperatures = [0.0, 0.7]
    
    # Generate runs (4 total)
    runs = []
    run_id = 0
    for model in models:
        for temp in temperatures:
            run_id += 1
            # Generate embedding (32-dim)
            embedding = [float(x) for x in np.random.uniform(-0.5, 0.5, 32)]
            
            # Generate hallucination risk (15-55)
            hallucination_risk = random.uniform(15, 55)
            
            # Generate reasons
            reasons = [f"Mock reason {run_id}"]
            
            # Generate logprob stats with small variations
            logprob_avg = random.uniform(-1.5, -0.5)
            logprob_std = random.uniform(0.1, 0.5)
            logprob_frac_low = random.uniform(0.1, 0.3)
            
            runs.append(ExperimentRun(
                model=model,
                temperature=temp,
                system_prompt=None,
                seed=None,
                output_text=f"Sample output for {model} at temperature {temp}",
                tokens=["Sample", "output", "for", model, "at", "temperature", str(temp)],
                logprob_avg=logprob_avg,
                logprob_std=logprob_std,
                logprob_frac_low=logprob_frac_low,
                embedding=embedding,
                latency_ms=random.uniform(250, 950),
                cost_usd=0.0,
                hallucination_risk=hallucination_risk,
                hallucination_reasons=reasons
            ))
    
    # Generate drift stats
    # Centroids per model (same dim)
    centroids = {}
    for model in models:
        centroids[model] = [float(x) for x in np.random.uniform(-0.5, 0.5, 32)]
    
    # Stability arrays (values ~0.90-0.99)
    stability = {}
    for model in models:
        stability[model] = [random.uniform(0.90, 0.99) for _ in range(2)]
    
    # by_temperature map with mean_risk and mean_stability
    by_temperature = {}
    for model in models:
        by_temperature[model] = {}
        for temp in temperatures:
            # mean_risk roughly increases with temperature
            mean_risk = 20 + (temp * 15)  # 20 at 0.0, 30.5 at 0.7
            mean_stability = random.uniform(0.90, 0.99)
            by_temperature[model][temp] = {"mean_risk": mean_risk, "mean_stability": mean_stability}
    
    # by_system_prompt (empty for mock)
    by_system_prompt = {model: {} for model in models}
    
    drift = DriftStats(
        centroid=centroids,
        stability=stability,
        by_temperature=by_temperature,
        by_system_prompt=by_system_prompt
    )
    
    return ExperimentResponse(runs=runs, drift=drift)

def get_adapter(model_name: str):
    name = model_name.lower()
    if name.startswith("gpt"):
        return OpenAIAdapter(model_name)
    if name.startswith("claude"):
        return AnthropicAdapter(model_name)
    # default stub
    return OpenAIAdapter(model_name)

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    if MOCK_MODE:
        return mock_compare_response(req)
    
    results = []
    for m in req.models:
        adapter = get_adapter(m)
        gen = adapter.generate(req.prompt, temperature=req.temperature, system_prompt=req.system_prompt)
        results.append(ModelResult(
            model=m,
            output_text=gen["output_text"],
            tokens=gen["tokens"],
            logprobs=gen.get("logprobs"),
            embedding=None
        ))

    texts = [r.output_text for r in results]
    labels = [r.model for r in results]
    vectors = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
    for r, vec in zip(results, vectors):
        r.embedding = vec

    sim = pairwise_similarity(labels, vectors)

    # Compute hallucination risk for each result
    for i, r in enumerate(results):
        neighbor_cos = []
        for j in range(len(results)):
            if j == i: continue
            neighbor_cos.append(sim[r.model][results[j].model])
        risk, reasons = compute_hallucination_risk(r.output_text, r.logprobs, neighbor_cos)
        r.hallucination_risk = risk
        r.hallucination_reasons = reasons

    # Compute pairwise diffs
    token_diffs = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            key = f"{results[i].model}||{results[j].model}"
            html_view = html_token_diff(results[i].tokens, results[j].tokens)
            uni_view = unified_token_diff(results[i].tokens, results[j].tokens, results[i].model, results[j].model)
            token_diffs[key] = {"html": html_view, "unified": uni_view}

# Compute logprob diffs for pairs where both have logprobs
    logprob_diffs = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1, r2 = results[i], results[j]
            if r1.logprobs is not None and r2.logprobs is not None:
                key = f"{r1.model}||{r2.model}"
                aligned = align_tokens_and_logprobs(r1.tokens, r1.logprobs, r2.tokens, r2.logprobs)
                summary = summarize_logprob_diff(aligned)
                if summary:
                    logprob_diffs[key] = summary
    # Update summaries with highest risk
    top = max(results, key=lambda x: x.hallucination_risk or 0.0)
    summaries = {
        "note": "Autopsy summary placeholder. Hook in heuristics later.",
        "risk_highlight": f"Highest estimated hallucination risk: {top.model} ({top.hallucination_risk:.1f}/100)."
    }
    return CompareResponse(results=results, embedding_similarity=sim, summaries=summaries, token_diffs=token_diffs, logprob_diffs=logprob_diffs)
    

# Helper functions for experiment endpoint
def lp_stats(lp):
    if not lp: return None, None, None
    arr = np.array(lp, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), float((arr < -2.5).mean())

def adapter_for(model):
    name = model.lower()
    if name.startswith("gpt"): return OpenAIAdapter(model)
    if name.startswith("claude"): return AnthropicAdapter(model)
    return OpenAIAdapter(model)

@app.post("/experiment", response_model=ExperimentResponse)
async def experiment(req: ExperimentRequest):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Received experiment request: {req}")
    logger.info(f"Request dict: {req.dict()}")
    
    if MOCK_MODE:
        return mock_experiment_response(req)
    
    seeds = req.seeds or [None]
    grid = []
    for m in req.models:
        for t in req.temperatures:
            for s in req.system_prompts:
                for sd in seeds:
                    grid.append((m, t, s, sd))

    # Run generation concurrently
    runs_raw = []
    def run_one(m, t, s, sd):
        adapter = adapter_for(m)
        start = time.perf_counter()
        gen = adapter.generate(req.prompt, temperature=t, system_prompt=s)
        latency_ms = (time.perf_counter() - start) * 1000.0
        avg, std, frac_low = lp_stats(gen.get("logprobs"))
        return {
            "model": m,
            "temperature": t,
            "system_prompt": s,
            "seed": sd,
            "output_text": gen["output_text"],
            "tokens": gen["tokens"],
            "logprob_avg": avg,
            "logprob_std": std,
            "logprob_frac_low": frac_low,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,  # placeholder
            "_logprobs": gen.get("logprobs")
        }

    # Note: adapters are synchronous. Run in thread pool via asyncio.to_thread
    runs_raw = [run_one(m, t, s, sd) for (m, t, s, sd) in grid]

    # Embeddings in batch for speed
    texts = [r["output_text"] for r in runs_raw]
    vectors = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
    for r, vec in zip(runs_raw, vectors):
        r["embedding"] = vec

    # Compute cross-model neighbor similarities for risk
    # For each run, neighbor list are other runs with different model
    # Precompute embeddings per model
    per_model_vecs = {}
    for r in runs_raw:
        per_model_vecs.setdefault(r["model"], []).append(r["embedding"])

    # Centroids
    centroids = {m: np.mean(np.array(vecs), axis=0).tolist() for m, vecs in per_model_vecs.items()}

    # Stability: cosine of each run vector to its model centroid
    def cos(a, b):
        a = np.array(a).reshape(1, -1)
        b = np.array(b).reshape(1, -1)
        return float(cosine_similarity(a, b)[0,0])

    # Build per-run hallucination risk using neighbor cosines to other models' centroids
    model_centroids = {m: np.array(v) for m, v in centroids.items()}

    runs = []
    stability = {m: [] for m in per_model_vecs.keys()}

    for r in runs_raw:
        neighbors = []
        for m, c in model_centroids.items():
            if m == r["model"]: continue
            neighbors.append(cos(r["embedding"], c))
        risk, reasons = compute_hallucination_risk(r["output_text"], r.get("_logprobs"), neighbors)
        stab = cos(r["embedding"], centroids[r["model"]])
        stability[r["model"]].append(stab)
        runs.append(ExperimentRun(
            model=r["model"],
            temperature=r["temperature"],
            system_prompt=r["system_prompt"],
            seed=r["seed"],
            output_text=r["output_text"],
            tokens=r["tokens"],
            logprob_avg=r["logprob_avg"],
            logprob_std=r["logprob_std"],
            logprob_frac_low=r["logprob_frac_low"],
            embedding=r["embedding"],
            latency_ms=r["latency_ms"],
            cost_usd=r["cost_usd"],
            hallucination_risk=risk,
            hallucination_reasons=reasons
        ))

    # Aggregations
    by_temperature = {m:{} for m in per_model_vecs.keys()}
    for m in per_model_vecs.keys():
        for t in req.temperatures:
            sel = [rr for rr in runs if rr.model == m and rr.temperature == t]
            if not sel: continue
            mean_risk = float(np.mean([rr.hallucination_risk for rr in sel]))
            mean_stab = float(np.mean([cos(rr.embedding, centroids[m]) for rr in sel]))
            by_temperature[m][t] = {"mean_risk": mean_risk, "mean_stability": mean_stab}

    by_system_prompt = {m:{} for m in per_model_vecs.keys()}
    for m in per_model_vecs.keys():
        for s in req.system_prompts:
            sel = [rr for rr in runs if rr.model == m and rr.system_prompt == s]
            if not sel: continue
            mean_risk = float(np.mean([rr.hallucination_risk for rr in sel]))
            mean_stab = float(np.mean([cos(rr.embedding, centroids[m]) for rr in sel]))
            # Handle None values properly - use "None" string as key for None values
            key = str(s) if s is not None else "None"
            by_system_prompt[m][key] = {"mean_risk": mean_risk, "mean_stability": mean_stab}

    drift = DriftStats(
        centroid={m: list(centroids[m]) for m in centroids},
        stability=stability,
        by_temperature=by_temperature,
        by_system_prompt=by_system_prompt
    )
    # Compute parameter influence
    runs_dicts = [r.dict() if hasattr(r, "dict") else r for r in runs]
    # baseline = lowest temperature and first system prompt present
    temps = sorted({r["temperature"] for r in runs_dicts})
    sysp = list({r["system_prompt"] for r in runs_dicts})
    temp_inf = compute_parameter_influence(runs_dicts, "temperature", temps[0] if temps else 0.0)
    sys_inf  = compute_parameter_influence(runs_dicts, "system_prompt", sysp[0] if sysp else None)

    # attach
    drift.parameter_influence = {"temperature": temp_inf, "system_prompt": sys_inf}
    return ExperimentResponse(runs=runs, drift=drift)