import numpy as np
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict, Any

def align_tokens_and_logprobs(tokens_a: List[str], lp_a: Optional[List[float]],
                              tokens_b: List[str], lp_b: Optional[List[float]]) -> List[Tuple[Optional[str], Optional[float], Optional[str], Optional[float], Optional[float]]]:
    if not lp_a or not lp_b:
        return []
    sm = SequenceMatcher(a=tokens_a, b=tokens_b)
    out = []
    ia, ib = 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                a_tok = tokens_a[i1 + k]; b_tok = tokens_b[j1 + k]
                a_lp = lp_a[i1 + k] if i1 + k < len(lp_a) else None
                b_lp = lp_b[j1 + k] if j1 + k < len(lp_b) else None
                out.append((a_tok, a_lp, b_tok, b_lp, (a_lp - b_lp) if (a_lp is not None and b_lp is not None) else None))
        else:
            # mark unmatched as insert/delete (no diff)
            for k in range(i2 - i1):
                a_tok = tokens_a[i1 + k]; a_lp = lp_a[i1 + k] if i1 + k < len(lp_a) else None
                out.append((a_tok, a_lp, None, None, None))
            for k in range(j2 - j1):
                b_tok = tokens_b[j1 + k]; b_lp = lp_b[j1 + k] if j1 + k < len(lp_b) else None
                out.append((None, None, b_tok, b_lp, None))
    return out

def summarize_logprob_diff(aligned):
    diffs = [abs(d) for *_, d in aligned if d is not None]
    if not diffs:
        return {}
    return {
        "mean_abs_diff": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)),
        "n_compared": int(len(diffs))
    }