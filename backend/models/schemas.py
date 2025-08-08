from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple

class CompareRequest(BaseModel):
    prompt: str
    models: List[str]
    temperature: Optional[float] = 0.2
    system_prompt: Optional[str] = None

class ModelResult(BaseModel):
    model: str
    output_text: str
    tokens: List[str]
    logprobs: Optional[List[float]] = None
    embedding: Optional[List[float]] = None
    hallucination_risk: Optional[float] = None
    hallucination_reasons: Optional[List[str]] = None

class CompareResponse(BaseModel):
    results: List[ModelResult]
    embedding_similarity: Dict[str, Dict[str, float]]
    summaries: Dict[str, Any]
    token_diffs: Dict[str, Dict[str, Any]] = {}

# Experiment models
class ExperimentRequest(BaseModel):
    prompt: str
    models: List[str]
    temperatures: List[float] = [0.2, 0.7]
    system_prompts: List[str] = [None]
    seeds: List[int] = []

class ExperimentRun(BaseModel):
    model: str
    temperature: float
    system_prompt: Optional[str]
    seed: Optional[int]
    output_text: str
    tokens: List[str]
    logprob_avg: Optional[float] = None
    logprob_std: Optional[float] = None
    logprob_frac_low: Optional[float] = None
    embedding: List[float]
    latency_ms: float
    cost_usd: float
    hallucination_risk: float
    hallucination_reasons: List[str]

class DriftStats(BaseModel):
    # keyed by model
    centroid: Dict[str, List[float]]
    stability: Dict[str, List[float]]  # cosine to centroid per run index aligned with runs list
    by_temperature: Dict[str, Dict[float, Dict[str, float]]]  # model -> temp -> {"mean_risk":..., "mean_stability":...}
    by_system_prompt: Dict[str, Dict[str, Dict[str, float]]]  # model -> sys -> {...}

class ExperimentResponse(BaseModel):
    runs: List[ExperimentRun]
    drift: DriftStats