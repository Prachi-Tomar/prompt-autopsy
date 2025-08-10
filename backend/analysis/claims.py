import re

# Very light heuristics: numbers, dates, "X is Y", capitalized entities
DATE_PAT = re.compile(r"\b(?:\d{4}|\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?)\b")
NUM_PAT  = re.compile(r"\b\d[\d,\.]*\b")
IS_PAT   = re.compile(r"\b([A-Z][a-zA-Z0-9][\w\s\-]{1,40})\s+(?:is|are|was|were)\s+([\w\s\-]{2,60})\b")

def extract_candidate_claims(text: str, max_claims: int = 12):
    claims = set()

    # numbers and dates sentences
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if NUM_PAT.search(sent) or DATE_PAT.search(sent):
            claims.add(sent.strip())

    # simple "X is Y" style
    for m in IS_PAT.finditer(text):
        span = m.group(0).strip()
        claims.add(span)

    # keep it small
    out = list(claims)
    out.sort(key=len)
    return out[:max_claims]