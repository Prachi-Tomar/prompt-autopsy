from typing import Tuple

# NOTE: These are placeholders. Update to match your accounts.
# Prices are per 1K tokens (USD).
OPENAI_PRICING = {
    "gpt-5":        (5.00, 15.00),   # (input, output)  <-- adjust
    "gpt-5-mini":   (0.60, 2.40),
    "gpt-4o":       (5.00, 15.00),
    "gpt-4o-mini":  (0.50, 1.50),
}
ANTHROPIC_PRICING = {
    "claude-3.5-sonnet-2024-10-22": (3.00, 15.00),
    "claude-3.5-sonnet-2024-06-20": (3.00, 15.00),
    "claude-3.5-haiku":             (0.80, 4.00),
    "claude-3-haiku":               (0.80, 4.00),
    # add others you use
}

def estimate_cost(model: str, provider: str, prompt_toks: int, completion_toks: int) -> float:
    table = OPENAI_PRICING if provider == "openai" else ANTHROPIC_PRICING
    # pick exact if present, else try prefix match
    if model in table:
        pin, pout = table[model]
    else:
        pin, pout = None, None
        for k, (ci, co) in table.items():
            if model.startswith(k.split(":")[0]):
                pin, pout = ci, co
                break
    if pin is None or pout is None:
        return 0.0
    return (prompt_toks / 1000.0) * pin + (completion_toks / 1000.0) * pout