import logging
from typing import Optional, Dict, Any, List
from .base import LLMAdapter
from ..utils.settings import GOOGLE_CLOUD_PROJECT, VERTEX_LOCATION, GEMINI_API_KEY, vertex_config_ok
from backend.analysis.pricing import estimate_cost
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Candidate, Part
import google.cloud.aiplatform as aiplatform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiVertexAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Initialize Vertex AI
        if vertex_config_ok():
            try:
                aiplatform.init(
                    project=GOOGLE_CLOUD_PROJECT,
                    location=VERTEX_LOCATION
                )
                self.model = GenerativeModel(model_name)
                logger.info(f"Vertex AI model {model_name} initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI model {model_name}: {e}")
                self.model = None
        else:
            self.model = None
            logger.warning(f"Vertex AI configuration not valid for model {model_name}")

    def generate(self, prompt: str, temperature: float = 0.2, system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response using Vertex AI Gemini.
        """
        # Use provided model or fall back to instance model
        model_to_use = model or self.model_name

        # Check if Vertex AI configuration is valid and model is initialized
        if not vertex_config_ok() or self.model is None:
            output_text = "[Gemini ERROR] Missing Vertex config: set GOOGLE_CLOUD_PROJECT and VERTEX_LOCATION (see README)."
            return {
                "model": model_to_use,
                "output_text": output_text,
                "tokens": None,
                "logprobs": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cost_usd": 0.0
            }
        
        try:
            # Prepare contents
            contents = []
            if system_prompt:
                contents.append(Part.from_text(system_prompt))
            contents.append(Part.from_text(prompt))
            
            # Generate content with logprobs
            response = self.model.generate_content(
                contents=contents,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    # Note: response_logprobs is not always available in all Vertex AI models
                    # We'll try to get logprobs if available
                ),
                stream=False
            )
            
            # Extract output text
            if response.candidates and response.candidates[0].content:
                output_text = response.candidates[0].content.parts[0].text
            else:
                output_text = "[Gemini ERROR] No response content received"
            
            # Simple tokenization (split by spaces) if no tokens from API
            tokens = output_text.split()
            
            # Try to extract logprobs if available
            logprobs = None
            # Note: Logprobs extraction depends on the specific model and API version
            # This is a simplified approach that may need adjustment based on actual API response
            
            # Extract token counts if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            
            # Try to get usage metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(response.usage_metadata, 'total_token_count', None)
            
            # If we don't have token counts from usage_metadata, estimate them
            if prompt_tokens is None:
                prompt_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
            if completion_tokens is None:
                completion_tokens = len(tokens)
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            
            # Compute cost
            cost = estimate_cost(model_to_use, "google", prompt_tokens or 0, completion_tokens or 0)
            cost_usd = round(cost, 6)
            
            logger.info(f"Vertex AI model {model_to_use} called successfully")
            
            return {
                "model": model_to_use,
                "output_text": output_text,
                "tokens": tokens,
                "logprobs": logprobs,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd
            }
            
        except Exception as e:
            error_msg = f"[Vertex AI ERROR: {type(e).__name__}] {e}"
            logger.error(f"Vertex AI API error: {str(e)}")
            return {
                "model": model_to_use,
                "output_text": error_msg,
                "tokens": error_msg.split(),
                "logprobs": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cost_usd": 0.0
            }