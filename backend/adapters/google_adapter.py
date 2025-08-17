import google.generativeai as genai
from typing import Optional, Dict, Any
from .base import LLMAdapter
from backend.utils.settings import GEMINI_API_KEY
from backend.analysis.pricing import estimate_cost
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    def generate(self, prompt: str, temperature: float = 0.2, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Check if model is configured
            if not self.model or not GEMINI_API_KEY:
                err = "[Google ERROR] No SDK or GEMINI_API_KEY configured."
                return {"output_text": err, "tokens": err.split(), "logprobs": None}

            # Prepare content
            content = []
            if system_prompt:
                content.append({"role": "user", "parts": [system_prompt]})
            content.append({"role": "user", "parts": [prompt]})

            # Make API call
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )

            # Extract content
            output_text = response.text if response.text else ""
            
            # Tokenize output (simple split for now)
            tokens = output_text.split()
            
            # Gemini does not currently return token-level logprobs
            logprobs = None
            
            # Extract usage information if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            
            # Try to get token usage from response
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', None)
                completion_tokens = getattr(usage, 'candidates_token_count', None)
                total_tokens = getattr(usage, 'total_token_count', None)
            
            # Compute cost
            cost = estimate_cost(self.model_name, "google", prompt_tokens or 0, completion_tokens or 0)
            
            return {
                "output_text": output_text,
                "tokens": tokens,
                "logprobs": logprobs,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost, 6)
            }
            
        except Exception as e:
            err = f"[Google ERROR: {type(e).__name__}] {e}"
            logger.error(f"Google API error: {str(e)}")
            return {
                "output_text": err,
                "tokens": err.split(),
                "logprobs": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cost_usd": None
            }