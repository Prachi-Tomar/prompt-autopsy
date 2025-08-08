from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Return dict with fields:
        - output_text: str
        - tokens: list[str] (placeholder tokenization ok)
        - logprobs: list[float] or None (placeholder ok)
        """
        pass