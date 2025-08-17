import logging
from typing import Optional, Dict, Any, List
from .base import LLMAdapter
from backend.utils import settings
from backend.analysis.pricing import estimate_cost

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

        # Models that don't support temperature parameter
        TEMP_LOCKED_MODELS = {"gpt-5", "gpt-5-mini", "gpt-4o-mini"}  # add more if needed

        use_temperature = None if self.model_name in TEMP_LOCKED_MODELS else temperature
        
        # Log when temperature is omitted
        if use_temperature is None and self.model_name in TEMP_LOCKED_MODELS:
            logging.info(f"Temperature parameter omitted for model {self.model_name} as it doesn't support custom temperature")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try Chat Completions first (works with gpt-4o, gpt-4o-mini, and typically gpt-5 if enabled)
        try:
            params = {
                "model": self.model_name,                 # e.g., "gpt-5", "gpt-5-mini", "gpt-4o"
                "messages": messages,
                # Logprobs are supported on many chat models; if not, API will ignore or error
                "logprobs": True,
                "top_logprobs": 5
            }
            if use_temperature is not None:
                params["temperature"] = use_temperature

            resp = _openai_client.chat.completions.create(**params)
            parsed = _extract_logprobs_from_chat(resp)
            
            # Extract usage information
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
            ct = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
            tt = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
            
            # Compute cost
            cost = estimate_cost(self.model_name, "openai", pt or 0, ct or 0)
            
            # Include usage and cost in return dict
            parsed.update({
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "cost_usd": round(cost, 6)
            })
            return parsed

        except Exception as e_chat:
            # Check if the error is related to temperature being unsupported
            if "temperature unsupported" in str(e_chat).lower():
                # Log that we're retrying without temperature
                logging.info(f"Temperature unsupported for model {self.model_name}, retrying without temperature parameter")
                
                # Retry without temperature parameter
                try:
                    params = {
                        "model": self.model_name,
                        "messages": messages,
                        # Logprobs are supported on many chat models; if not, API will ignore or error
                        "logprobs": True,
                        "top_logprobs": 5
                    }
                    
                    resp = _openai_client.chat.completions.create(**params)
                    parsed = _extract_logprobs_from_chat(resp)
                    
                    # Extract usage information
                    usage = getattr(resp, "usage", None)
                    pt = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
                    ct = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
                    tt = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
                    
                    # Compute cost
                    cost = estimate_cost(self.model_name, "openai", pt or 0, ct or 0)
                    
                    # Include usage and cost in return dict
                    parsed.update({
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt,
                        "cost_usd": round(cost, 6)
                    })
                    return parsed
                except Exception as e_retry:
                    # If retry also fails, fall back to no logprobs
                    pass
            
            # If the model is missing or logprobs unsupported, retry without logprobs
            try:
                params = {
                    "model": self.model_name,
                    "messages": messages
                }
                if use_temperature is not None:
                    params["temperature"] = use_temperature

                resp = _openai_client.chat.completions.create(**params)
                parsed = _extract_logprobs_from_chat(resp)
                # Ensure we mark logprobs as None in this path
                parsed["logprobs"] = None
                
                # Extract usage information
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
                ct = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
                tt = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
                
                # Compute cost
                cost = estimate_cost(self.model_name, "openai", pt or 0, ct or 0)
                
                # Include usage and cost in return dict
                parsed.update({
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                    "cost_usd": round(cost, 6)
                })
                return parsed
            except Exception as e_final:
                # If both temperature and logprobs are unsupported, we still need to return a response
                # Retry without either temperature or logprobs
                try:
                    params = {
                        "model": self.model_name,
                        "messages": messages
                    }
                    
                    resp = _openai_client.chat.completions.create(**params)
                    parsed = _extract_logprobs_from_chat(resp)
                    # Ensure we mark logprobs as None in this path
                    parsed["logprobs"] = None
                    
                    # Extract usage information
                    usage = getattr(resp, "usage", None)
                    pt = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
                    ct = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
                    tt = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
                    
                    # Compute cost
                    cost = estimate_cost(self.model_name, "openai", pt or 0, ct or 0)
                    
                    # Include usage and cost in return dict
                    parsed.update({
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt,
                        "cost_usd": round(cost, 6)
                    })
                    return parsed
                except Exception as e_last:
                    err = f"[OpenAI ERROR: {type(e_last).__name__}] {e_last}"
                    return {"output_text": err, "tokens": err.split(), "logprobs": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "cost_usd": None}