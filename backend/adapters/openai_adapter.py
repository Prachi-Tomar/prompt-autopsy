import openai
from typing import Optional, Dict, Any, List
from .base import LLMAdapter
from ..utils.settings import OPENAI_API_KEY
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, prompt: str, temperature: float = 0.2, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Determine if logprobs are supported
            logprobs_supported = self.model_name in ["gpt-4", "gpt-4o"]
            
            # Prepare parameters for API call
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Add logprobs if supported
            if logprobs_supported:
                api_params["logprobs"] = True
                api_params["top_logprobs"] = 1
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            # Extract content
            output_text = response.choices[0].message.content
            
            # Log token count
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"OpenAI {self.model_name} - Tokens: {response.usage.total_tokens}")
            
            # Tokenize output
            tokens = output_text.split()
            
            # Extract logprobs if available
            logprobs_list = None
            if logprobs_supported and response.choices[0].logprobs:
                logprobs_list = []
                for content in response.choices[0].logprobs.content:
                    if content.top_logprobs:
                        logprobs_list.append(content.top_logprobs[0].logprob)
                    else:
                        logprobs_list.append(None)
            
            return {
                "output_text": output_text,
                "tokens": tokens,
                "logprobs": logprobs_list
            }
            
        except Exception as e:
            error_msg = f"[ERROR: {str(e)}]"
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "output_text": error_msg,
                "tokens": error_msg.split(),
                "logprobs": None
            }