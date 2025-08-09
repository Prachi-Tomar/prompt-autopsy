import anthropic
from typing import Optional, Dict, Any
from .base import LLMAdapter
from ..utils.settings import ANTHROPIC_API_KEY
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnthropicAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, prompt: str, temperature: float = 0.2, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1024  # Required parameter for Anthropic
            )
            
            # Extract content
            output_text = response.content[0].text
            
            # Log token count
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Anthropic {self.model_name} - Tokens: {response.usage.output_tokens + response.usage.input_tokens}")
            
            # Tokenize output (simple split for now)
            tokens = output_text.split()
            
            # Anthropic does not currently return token-level logprobs
            logprobs = None
            
            return {
                "output_text": output_text,
                "tokens": tokens,
                "logprobs": logprobs
            }
            
        except Exception as e:
            err = f"[Anthropic ERROR: {type(e).__name__}] {e}"
            logger.error(f"Anthropic API error: {str(e)}")
            return {
                "output_text": err,
                "tokens": err.split(),
                "logprobs": None
            }