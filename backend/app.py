from fastapi import FastAPI
from backend.models.schemas import CompareRequest, CompareResponse, ModelResult
from backend.adapters.openai_adapter import OpenAIAdapter
from backend.adapters.anthropic_adapter import AnthropicAdapter
from backend.analysis.embeddings import embed_texts, pairwise_similarity
from backend.analysis.hallucination import compute_hallucination_risk
from backend.analysis.diff import html_token_diff, unified_token_diff

app = FastAPI(title="Prompt Autopsy API")

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

    # Update summaries with highest risk
    top = max(results, key=lambda x: x.hallucination_risk or 0.0)
    summaries = {
        "note": "Autopsy summary placeholder. Hook in heuristics later.",
        "risk_highlight": f"Highest estimated hallucination risk: {top.model} ({top.hallucination_risk:.1f}/100)."
    }
    return CompareResponse(results=results, embedding_similarity=sim, summaries=summaries, token_diffs=token_diffs)

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
    seeds = req.seeds or [None]
    grid = []
    for m in req.models:
        for t in req.temperatures:
            for s in req.system_prompts:
                for sd in seeds:
                    grid.append((m, t, s, sd))

    # Run generation concurrently
    runs_raw = []
    async def run_one(m, t, s, sd):
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
    runs_raw = await asyncio.gather(*[asyncio.to_thread(run_one, m,t,s,sd) for (m,t,s,sd) in grid])

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
            by_system_prompt[m][str(s)] = {"mean_risk": mean_risk, "mean_stability": mean_stab}

    drift = DriftStats(
        centroid={m: list(centroids[m]) for m in centroids},
        stability=stability,
        by_temperature=by_temperature,
        by_system_prompt=by_system_prompt
    )
    return ExperimentResponse(runs=runs, drift=drift)