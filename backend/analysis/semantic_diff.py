import re
import math
from collections import Counter
from typing import Dict, Tuple

NEG_WORDS = {"no","not","never","none","nobody","nothing","nowhere","neither","nor","cannot","can't","won't","doesn't","isn't","aren't","didn't","haven't","hasn't","hadn't","without"}
HEDGE = {"might","may","could","possibly","likely","appears","seems","reportedly"}
POS = {"great","good","excellent","beneficial","positive","success","improve","advantage"}
NEG = {"bad","poor","terrible","negative","harm","risk","fail","worse","disadvantage"}

def _tokens(s: str):
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def _ngrams(tokens, n=2):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _jaccard(a: set, b: set):
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def _keyphrases(text: str):
    toks = _tokens(text)
    # crude keyphrase proxy: bigrams + capitalized words
    caps = re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", text)
    return set(_ngrams(toks,2)) | set([c.lower() for c in caps])

def _sentiment(tokens):
    pos = sum(t in POS for t in tokens)
    neg = sum(t in NEG for t in tokens)
    total = pos + neg
    if total == 0: return 0.0
    return (pos - neg) / total  # -1..+1

def classify_semantic_divergence(a_text: str, b_text: str) -> Dict[str, float]:
    """
    Returns scores in 0..1 (higher = stronger evidence):
      style, omission, contradiction, ordering, tone
    Heuristics only; cheap & explainable.
    """
    ta, tb = _tokens(a_text), _tokens(b_text)

    # STYLE: length & hedge difference
    len_diff = abs(len(ta) - len(tb)) / max(1, (len(ta)+len(tb))/2)
    hedge_gap = abs(sum(w in HEDGE for w in ta) - sum(w in HEDGE for w in tb)) / max(1, len(ta)+len(tb))
    style = min(1.0, 0.6*len_diff + 8.0*hedge_gap)

    # OMISSION: keyphrase overlap (low overlap => higher omission)
    ka, kb = _keyphrases(a_text), _keyphrases(b_text)
    kp_jacc = _jaccard(ka, kb)
    omission = 1.0 - kp_jacc  # less shared keyphrases → more omission risk

    # CONTRADICTION: negation over shared unigrams
    shared = set(ta) & set(tb)
    neg_a = sum(w in NEG_WORDS for w in ta if w in shared)
    neg_b = sum(w in NEG_WORDS for w in tb if w in shared)
    contradiction = min(1.0, abs(neg_a - neg_b) / max(1, len(shared)))

    # ORDERING: compare bigram vs unigram overlap
    big_a, big_b = set(_ngrams(ta,2)), set(_ngrams(tb,2))
    uni_j = _jaccard(set(ta), set(tb))
    bi_j  = _jaccard(big_a, big_b)
    ordering = max(0.0, uni_j - bi_j)  # if unigrams match but bigrams don't → reordering

    # TONE: sentiment gap magnitude
    tone = min(1.0, abs(_sentiment(ta) - _sentiment(tb)))

    return {
        "style": float(style),
        "omission": float(omission),
        "contradiction": float(contradiction),
        "ordering": float(ordering),
        "tone": float(tone)
    }