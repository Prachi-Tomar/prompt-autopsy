import re
import numpy as np

HEDGING = [
    "might","may","could","likely","possibly","reportedly",
    "unconfirmed","i think","i believe","it seems","appears"
]
REF_PATTERNS = [
    r"https?://", r"doi:\s*\S+", r"\[\d+\]"
]
PROPER_LIKE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")

def text_stats(text: str):
    lower = text.lower()
    hedging_count = sum(lower.count(h) for h in HEDGING)
    has_reference = any(re.search(p, text) for p in REF_PATTERNS)
    numeric_token_count = len(re.findall(r"\b\d[\d,\.]*\b", text))
    proper_noun_like_count = len(PROPER_LIKE.findall(text))
    return hedging_count, has_reference, numeric_token_count, proper_noun_like_count

def logprob_stats(logprobs):
    if not logprobs:
        return None
    arr = np.array(logprobs, dtype=float)
    avg = float(np.mean(arr))
    std = float(np.std(arr))
    frac_low = float((arr < -2.5).mean()) if arr.size else 0.0
    return {"avg": avg, "std": std, "frac_low": frac_low}

def compute_divergence_penalty(min_cosine: float):
    # Penalize when responses disagree strongly
    # 0.80 -> +8, 0.90 -> +4, 0.95 -> +1, >=0.97 -> 0
    if min_cosine >= 0.97: return 0.0
    if min_cosine >= 0.95: return 1.0
    if min_cosine >= 0.90: return 4.0
    if min_cosine >= 0.80: return 8.0
    return 8.0

def clamp(x, lo, hi): return max(lo, min(hi, x))

def compute_hallucination_risk(output_text: str, logprobs, neighbor_cosines: list[float]):
    hedging_count, has_reference, numeric_count, proper_like = text_stats(output_text)
    lp = logprob_stats(logprobs) if logprobs is not None else None

    base = 0.0
    reasons = []

    if lp:
        base_lp = clamp(50.0 * lp["frac_low"], 0, 50)
        base += base_lp
        if base_lp > 0: reasons.append(("Low-confidence tokens", base_lp))
        instab = clamp(10.0 * lp["std"], 0, 15)
        base += instab
        if instab > 0: reasons.append(("High token-level variance", instab))
        if lp["avg"] < -1.5:
            base += 10.0
            reasons.append(("Low average logprob", 10.0))
    else:
        # No logprobs: small baseline uncertainty
        base += 8.0
        reasons.append(("No token-level confidence available", 8.0))

    hedging_penalty = clamp(2.0 * hedging_count, 0, 12)
    if hedging_penalty > 0: reasons.append(("Hedging/speculative language", hedging_penalty))
    spec_penalty = clamp(0.5 * numeric_count + 0.5 * proper_like, 0, 12)
    if spec_penalty > 0: reasons.append(("Specifics without references", spec_penalty))
    ref_credit = -8.0 if has_reference else 0.0
    if ref_credit < 0: reasons.append(("Has references/citations", ref_credit))

    min_cos = min(neighbor_cosines) if neighbor_cosines else 1.0
    div_pen = compute_divergence_penalty(min_cos)
    if div_pen > 0: reasons.append(("Disagrees with other models", div_pen))

    risk = clamp(base + hedging_penalty + spec_penalty + ref_credit + div_pen, 0, 100)

    # Keep top 3 reasons by absolute impact
    reasons = sorted(reasons, key=lambda x: abs(x[1]), reverse=True)[:3]
    reasons_out = [f"{label} ({'+' if val>=0 else ''}{val:.1f})" for label, val in reasons]
    return float(risk), reasons_out