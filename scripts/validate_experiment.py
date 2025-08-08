import sys
import json

# Load the response from stdin
data = json.load(sys.stdin)

# Validate the response structure
assert "runs" in data and isinstance(data["runs"], list) and data["runs"], "Missing runs"
assert "drift" in data and isinstance(data["drift"], dict), "Missing drift"

# Validate the drift structure
for k in ("centroid","stability","by_temperature","by_system_prompt"):
    assert k in data["drift"], f"Missing drift.{k}"

print("OK: /experiment schema looks good")