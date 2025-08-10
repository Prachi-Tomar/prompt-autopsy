import requests, re, html
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from backend.analysis.claims import extract_candidate_claims

WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

@lru_cache(maxsize=512)
def _wiki_search(query: str) -> Optional[str]:
    try:
        r = requests.get(WIKI_SEARCH, params={
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json"
        }, timeout=6)
        r.raise_for_status()
        data = r.json()
        if data and len(data) >= 2 and data[1]:
            return data[1][0]
    except Exception:
        return None
    return None

@lru_cache(maxsize=512)
def _wiki_summary(title: str) -> Optional[str]:
    try:
        r = requests.get(WIKI_SUMMARY.format(title=title.replace(" ", "_")), timeout=6)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        # prefer extract (plain text)
        txt = data.get("extract") or ""
        return _clean(txt)
    except Exception:
        return None

def verify_claims(claims: List[str]) -> Dict[str, List[Dict]]:
    results = []
    for c in claims:
        q = c[:120]
        title = _wiki_search(q) or _wiki_search(c.split(".")[0][:80] if "." in c else c[:80])
        if not title:
            results.append({"claim": c, "verdict": "ambiguous", "evidence": None})
            continue
        summary = _wiki_summary(title)
        if not summary:
            results.append({"claim": c, "verdict": "ambiguous", "evidence": title})
            continue

        # crude verdict: if most non-stopword tokens appear in summary → supported
        toks = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", c) if len(t) > 2]
        summ = summary.lower()
        hit = sum(t in summ for t in toks)
        ratio = hit / max(1, len(toks))
        if ratio >= 0.55:
            verd = "supported"
        elif ratio <= 0.25:
            verd = "unsupported"
        else:
            verd = "ambiguous"
        results.append({"claim": c, "verdict": verd, "evidence": title})
    # aggregate
    agg = {"supported":0,"unsupported":0,"ambiguous":0}
    for r in results: agg[r["verdict"]] += 1
    return {"supported": agg["supported"], "unsupported": agg["unsupported"], "ambiguous": agg["ambiguous"], "claims": results}

def analyze_factcheck(text: str) -> Dict[str, List[Dict]]:
    """
    Analyze text for fact-checking using Wikipedia API.
    Returns a dict with counts and claims details.
    """
    claims = extract_candidate_claims(text)
    return verify_claims(claims)