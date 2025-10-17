# search/services/query_parser.py
from .llm_service import huggingface_service


class AdvancedQueryProcessor:
    def __init__(self):
        self.llm_service = huggingface_service
        # Keep your existing stop words and basic logic

    def parse_query(self, query: str) -> dict:
        """
        Enhanced parsing with Hugging Face understanding
        """
        # Get deep analysis from Hugging Face
        hf_analysis = self.llm_service.analyze_research_query(query)

        # Enhanced with your reliable basic parser
        basic_analysis = self._basic_parse(query)

        return {
            **hf_analysis,
            **basic_analysis,
            "original_query": query
        }

    def _basic_parse(self, query: str) -> dict:
        """Your existing, reliable parsing logic"""
        # Copy your current parse_query method here
        # This ensures it always works even if HF is slow/down