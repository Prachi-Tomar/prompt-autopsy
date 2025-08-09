from typing import Optional, Dict, Any, List
from .base import LLMAdapter
from backend.utils import settings

# OpenAI SDK v1.x
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY or None)
except Exception:
    _openai_client = None

def _extract_logprobs_from_chat(resp) -> Dict[str, Any]:
    """
    Extracts output_text, tokens, and token logprobs from Chat Completions response (SDK v1.x).
    Returns dict with keys: output_text, tokens, logprobs (or None).
    """
    try:
        choice = resp.choices[0]
        text = choice.message.content or ""
        tokens: List[str] = []
        lps: List[float] = []

        # Newer SDKs expose choice.logprobs.content as a list of items with .token and .logprob
        lp_content = getattr(choice, "logprobs", None)
        if lp_content and getattr(lp_content, "content", None):
            for item in lp_content.content:
                tok = getattr(item, "token", None)
                lp = getattr(item, "logprob", None)
                if tok is not None:
                    tokens.append(tok)
                    lps.append(float(lp) if lp is not None else None)

        # Fallback: if no per-token info, at least return text tokens
        if not tokens:
            tokens = text.split()
            lps = []  # logprobs unsupported

        return {"output_text": text, "tokens": tokens, "logprobs": lps if lps else None}
    except Exception:
        # Extremely defensive fallback
        txt = resp.choices[0].message.content if resp and resp.choices else ""
        return {"output_text": txt, "tokens": txt.split(), "logprobs": None}

class OpenAIAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.2, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if not _openai_client or not settings.OPENAI_API_KEY:
            err = "[OpenAI ERROR] No SDK or OPENAI_API_KEY configured."
            return {"output_text": err, "tokens": err.split(), "logprobs": None}

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try Chat Completions first (works with gpt-4o, gpt-4o-mini, and typically gpt-5 if enabled)
        try:
            resp = _openai_client.chat.completions.create(
                model=self.model_name,                 # e.g., "gpt-5", "gpt-5-mini", "gpt-4o"
                messages=messages,
                temperature=temperature,
                # Logprobs are supported on many chat models; if not, API will ignore or error
                logprobs=True,
                top_logprobs=5
            )
            parsed = _extract_logprobs_from_chat(resp)
            return parsed

        except Exception as e_chat:
            # If the model is missing or logprobs unsupported, retry without logprobs
            try:
                resp = _openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                parsed = _extract_logprobs_from_chat(resp)
                # Ensure we mark logprobs as None in this path
                parsed["logprobs"] = None
                return parsed
            except Exception as e_final:
                err = f"[OpenAI ERROR: {type(e_final).__name__}] {e_final}"
                return {"output_text": err, "tokens": err.split(), "logprobs": None}