import sys
import json

# Load the response from stdin
data = json.load(sys.stdin)

# Validate the response structure
assert "results" in data and isinstance(data["results"], list) and data["results"], "Missing results"
assert "embedding_similarity" in data, "Missing similarity"
assert "summaries" in data, "Missing summaries"
assert "token_diffs" in data, "Missing token_diffs"

# Validate the first result
r = data["results"][0]
for k in ("model","output_text","tokens","hallucination_risk"):
    assert k in r, f"Missing result.{k}"

print("OK: /compare schema looks good")