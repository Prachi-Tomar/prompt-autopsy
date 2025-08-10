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
from backend.analysis.semantic_diff import classify_semantic_divergence
from backend.analysis.factcheck import analyze_factcheck
from backend.analysis.claims import extract_candidate_claims
from backend.analysis.factcheck import verify_claims
import os

FACTCHECK_ON = os.getenv("FACTCHECK", "0") == "1"

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
            hallucination_reasons=reasons,
            factcheck={
                "supported": 5,
                "unsupported": 2,
                "ambiguous": 3,
                "claims": [
                    {"claim": "Quantum computing uses qubits", "verdict": "supported", "evidence": "Well-established fact"},
                    {"claim": "qubits can be both 0 and 1 at once", "verdict": "supported", "evidence": "Quantum superposition principle"}
                ]
            } if FACTCHECK_ON else None
        ))
    
    # Create embedding similarity matrix (2x2 with 1.0 diagonal and ~0.90 off-diagonal)
    embedding_similarity = {
        "gpt-4o": {"gpt-4o": 1.0, "claude-3-opus": 0.91},
        "claude-3-opus": {"gpt-4o": 0.91, "claude-3-opus": 1.0}
    }
    
    # Summaries with risk_highlight
    summaries = {
        "note": "This is a mock response for demonstration purposes.",
        "risk_highlight": "Highest estimated hallucination risk: claude-3-opus (28.7/100).",
        "factcheck_summary": {
            "supported": 10,
            "unsupported": 4,
            "ambiguous": 6
        } if FACTCHECK_ON else None
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

    # Compute semantic divergence for each pair
    semantic_diffs = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            scores = classify_semantic_divergence(a.output_text or "", b.output_text or "")
            semantic_diffs[f"{a.model}||{b.model}"] = scores

    return CompareResponse(
        results=results,
        embedding_similarity=embedding_similarity,
        summaries=summaries,
        token_diffs=token_diffs,
        logprob_diffs=logprob_diffs,
        semantic_diffs=semantic_diffs,
        factcheck_summary=summaries.get("factcheck_summary")
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

# Model alias mapping for robustness to model ID changes
MODEL_ALIASES = {
    # OpenAI old → current
    "gpt-4": "gpt-4o",
    "gpt4": "gpt-4o",
    "gpt-4-turbo": "gpt-4o",
    "gpt-3.5-turbo": "gpt-4o-mini",

    # GPT-5
    "gpt5": "gpt-5",
    "gpt-5-turbo": "gpt-5",

    # Anthropic friendly → exact IDs (update if your dashboard shows different date codes)
    "claude-3.5-sonnet-2024-10-22": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet-2024-06-20": "claude-3-5-sonnet-20240620",
    "claude-3.5-haiku": "claude-3-5-haiku-20240307",
    "claude-3-haiku": "claude-3-haiku-20240307"
}

def get_adapter(model_name: str):
    raw = model_name
    name = model_name.strip()
    key = name.lower()
    for k, v in MODEL_ALIASES.items():
        if key == k.lower():
            name = v
            break

    lname = name.lower()
    if lname.startswith("gpt"):
        return OpenAIAdapter(name)
    if lname.startswith("claude"):
        return AnthropicAdapter(name)
    return OpenAIAdapter(name)  # fallback

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
        
        # Add fact-check data
        if FACTCHECK_ON:
            cands = extract_candidate_claims(r.output_text or "")
            r.factcheck = verify_claims(cands)
        else:
            r.factcheck = None

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
                
    # Compute semantic divergence for each pair
    semantic_diffs = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1, r2 = results[i], results[j]
            key = f"{r1.model}||{r2.model}"
            semantic_diffs[key] = classify_semantic_divergence(r1.output_text, r2.output_text)
                
    # Compute aligned token data for pairs where both have logprobs
    aligned_data = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            if a.logprobs is not None and b.logprobs is not None:
                aligned = align_tokens_and_logprobs(a.tokens, a.logprobs, b.tokens, b.logprobs)
                key = f"{a.model}||{b.model}"
                aligned_data[key] = aligned
            
    # Update summaries with computed narrative
    summaries = build_autopsy_summary(results, sim, logprob_diffs)
    
    # Also build an aggregate for CompareResponse:
    if FACTCHECK_ON:
        agg = {"supported":0,"unsupported":0,"ambiguous":0}
        for r in results:
            if r.factcheck:
                agg["supported"] += r.factcheck["supported"]
                agg["unsupported"] += r.factcheck["unsupported"]
                agg["ambiguous"] += r.factcheck["ambiguous"]
        fact_summary = agg
    else:
        fact_summary = None
        
    return CompareResponse(results=results, embedding_similarity=sim, summaries=summaries, token_diffs=token_diffs, logprob_diffs=logprob_diffs, semantic_diffs=semantic_diffs, aligned_logprobs=aligned_data, factcheck_summary=fact_summary)
    

# Helper functions for experiment endpoint
def lp_stats(lp):
    if not lp: return None, None, None
    arr = np.array(lp, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), float((arr < -2.5).mean())

def build_autopsy_summary(results, sim_matrix, logprob_diffs):
    """
    results: list[ModelResult]
    sim_matrix: dict[str, dict[str, float]]
    logprob_diffs: dict["A||B" -> {mean_abs_diff, max_abs_diff, n_compared}]
    Returns dict with keys:
      summary (short paragraph),
      highlights (list[str]),
      top_model (str),
      highest_risk (str, float),
      closest_pair (tuple[str,str,float]),
      farthest_pair (tuple[str,str,float])
    """
    if not results:
        return {"summary": "No results.", "highlights": []}

    # Highest risk
    top = max(results, key=lambda r: (r.hallucination_risk or 0))
    highest_risk = (top.model, float(top.hallucination_risk or 0.0))

    # Similarity extremes
    labels = [r.model for r in results]
    closest = (None, None, -1.0)
    farthest = (None, None, 2.0)
    for a in labels:
        for b in labels:
            if a >= b:  # avoid dup/self
                continue
            s = sim_matrix.get(a, {}).get(b, 0.0)
            if s > closest[2]:
                closest = (a, b, float(s))
            if s < farthest[2]:
                farthest = (a, b, float(s))

    # Logprob divergence (if available)
    lp_note = None
    if logprob_diffs:
        pair, stats = max(logprob_diffs.items(), key=lambda kv: kv[1].get("mean_abs_diff", 0.0))
        lp_note = f"Largest confidence gap: {pair} (mean |Δlogprob| {stats['mean_abs_diff']:.2f})."

    # Compose
    highlights = []
    highlights.append(f"Highest estimated hallucination risk: {highest_risk[0]} ({highest_risk[1]:.1f}/100).")
    if closest[0]:
        highlights.append(f"Closest pair by meaning: {closest[0]} vs {closest[1]} (cos {closest[2]:.3f}).")
    if farthest[0]:
        highlights.append(f"Most divergent pair: {farthest[0]} vs {farthest[1]} (cos {farthest[2]:.3f}).")
    if lp_note:
        highlights.append(lp_note)

    # Influence detection (simple)
    if len(results) == 2:
        r1, r2 = results
        diff_temp = (r1.temperature != r2.temperature) if hasattr(r1, "temperature") and hasattr(r2, "temperature") else False
        diff_sys = (r1.system_prompt != r2.system_prompt) if hasattr(r1, "system_prompt") and hasattr(r2, "system_prompt") else False
        if diff_temp or diff_sys:
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([r1.embedding], [r2.embedding])[0,0] if r1.embedding and r2.embedding else None
            risk_delta = (r2.hallucination_risk or 0) - (r1.hallucination_risk or 0)
            if diff_temp:
                summaries_note = f"Changing temperature from {r1.temperature} to {r2.temperature} changed similarity to {sim:.3f} and risk by {risk_delta:+.1f}."
            elif diff_sys:
                summaries_note = f"Changing system prompt altered similarity to {sim:.3f} and risk by {risk_delta:+.1f}."
            else:
                summaries_note = None
            if summaries_note:
                highlights.append(summaries_note)
                influence_sentence = summaries_note
            else:
                influence_sentence = None
        else:
            influence_sentence = None
    else:
        influence_sentence = None

    # Short paragraph
    parts = []
    parts.append(f"{len(results)} models compared.")
    parts.append(highlights[0])
    if closest[0]:
        parts.append(f"{closest[0]} and {closest[1]} were most semantically aligned.")
    if farthest[0]:
        parts.append(f"{farthest[0]} and {farthest[1]} differed the most.")
    if lp_note:
        parts.append("Confidence varied across models; see logprob differences for details.")
    summary = " ".join(parts)
    
    # Aggregate fact-check results for summary
    if FACTCHECK_ON:
        total_supported = sum(r.factcheck.get("supported", 0) if r.factcheck else 0 for r in results)
        total_unsupported = sum(r.factcheck.get("unsupported", 0) if r.factcheck else 0 for r in results)
        total_ambiguous = sum(r.factcheck.get("ambiguous", 0) if r.factcheck else 0 for r in results)
        
        factcheck_summary = {
            "supported": total_supported,
            "unsupported": total_unsupported,
            "ambiguous": total_ambiguous
        }
    else:
        factcheck_summary = None

    # Choose a 'top_model' heuristic: lowest risk, highest similarity to group centroid
    import numpy as np
    vecs = [(r.model, np.array(r.embedding)) for r in results if r.embedding is not None]
    top_model = highest_risk[0]
    if vecs:
        centroid = np.mean([v for _, v in vecs], axis=0, keepdims=True)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = [(m, float(cosine_similarity(v.reshape(1,-1), centroid)[0,0])) for m, v in vecs]
        # prefer low risk then high centroid similarity
        def score(r):
            risk = r.hallucination_risk or 0.0
            sim = next((s for m,s in sims if m == r.model), 0.0)
            return (-risk, sim)
        best = max(results, key=score)
        top_model = best.model

    return {
        "summary": summary,
        "highlights": highlights,
        "top_model": top_model,
        "highest_risk": {"model": highest_risk[0], "score": highest_risk[1]},
        "closest_pair": {"a": closest[0], "b": closest[1], "cosine": closest[2]},
        "farthest_pair": {"a": farthest[0], "b": farthest[1], "cosine": farthest[2]},
        "influence_sentence": influence_sentence,
        "factcheck_summary": factcheck_summary
    }

def adapter_for(model):
    raw = model
    name = model.strip()
    key = name.lower()
    for k, v in MODEL_ALIASES.items():
        if key == k.lower():
            name = v
            break

    lname = name.lower()
    if lname.startswith("gpt"):
        return OpenAIAdapter(name)
    if lname.startswith("claude"):
        return AnthropicAdapter(name)
    return OpenAIAdapter(name)  # fallback

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