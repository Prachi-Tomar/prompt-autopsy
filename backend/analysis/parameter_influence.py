import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List

def compute_parameter_influence(runs: List[dict], group_key: str, baseline_value):
    """
    Returns dict[model] = { value: {'similarity_to_baseline': float, 'risk_change': float} }
    """
    out: Dict[str, Dict[Any, Dict[str, float]]] = {}
    models = sorted({r['model'] for r in runs})
    for m in models:
        mruns = [r for r in runs if r['model'] == m]
        base_runs = [r for r in mruns if r[group_key] == baseline_value]
        if not base_runs:
            continue
        base_vecs = np.array([r['embedding'] for r in base_runs])
        base_centroid = base_vecs.mean(axis=0, keepdims=True)
        base_risk = float(np.mean([r['hallucination_risk'] for r in base_runs]))
        out[m] = {}
        values = sorted({r[group_key] for r in mruns}, key=lambda x: (str(x)))
        for v in values:
            vruns = [r for r in mruns if r[group_key] == v]
            if not vruns:
                continue
            vecs = np.array([r['embedding'] for r in vruns])
            sims = cosine_similarity(vecs, base_centroid)
            risk = float(np.mean([r['hallucination_risk'] for r in vruns]))
            out[m][v] = {
                "similarity_to_baseline": float(np.mean(sims)),
                "risk_change": float(risk - base_risk),
            }
    return out