# search/services/llm_service.py
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceService:
    def __init__(self):
        self.models = {
            'intent': "facebook/bart-large-mnli",  # Zero-shot classification
            'ner': "dslim/bert-base-NER",  # Named Entity Recognition
            'summarization': "facebook/bart-large-cnn",  # Summarization
            'embedding': "sentence-transformers/all-MiniLM-L6-v2"  # Similarity
        }

        self.api_token = os.getenv('HF_API_TOKEN')

    def analyze_research_query(self, query: str) -> dict:
        """
        Multi-step analysis using specialized models
        """
        # Step 1: Intent classification
        intent = self._classify_intent(query)

        # Step 2: Extract entities/keywords
        entities = self._extract_entities(query)

        # Step 3: Basic structure parsing (your existing logic)
        basic_analysis = self._basic_parse(query)

        return {
            "intent": intent,
            "entities": entities,
            **basic_analysis,
            "original_query": query,
            "analysis_method": "huggingface_multi_model"
        }

    def _classify_intent(self, query: str) -> dict:
        """
        Use zero-shot classification to understand query type
        """
        url = f"https://api-inference.huggingface.co/models/{self.models['intent']}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        # Common research query types
        candidate_labels = [
            "find specific papers",    # Targeted search
            "literature review",       # Broad overview
            "empirical studies",       # Experimental work
            "theoretical research",    # Mathematical/conceptual
            "recent developments",     # Latest papers
            "highly cited papers",     # Influential work
            "practical applications",  # Real-world use cases
            "comprehensive survey"     # Systematic review
]

        payload = {
            "inputs": query,
            "parameters": {"candidate_labels": candidate_labels}
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return {
                    "primary_intent": result['labels'][0],
                    "confidence": result['scores'][0],
                    "all_intents": list(zip(result['labels'], result['scores']))
                }
        except (requests.RequestException, ValueError, KeyError, Exception) as e:
            # Log for debugging
            print(f"Error in _classify_intent: {e}")

        return {"primary_intent": "find specific papers", "confidence": 0.8}

    def _extract_entities(self, query: str) -> list:
        """
        Extract research-related entities using NER
        """
        url = f"https://api-inference.huggingface.co/models/{self.models['ner']}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        payload = {"inputs": query}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                entities = response.json()
                # Filter for relevant entities (topics, methods, etc.)
                relevant_entities = [
                    entity for entity in entities
                    if entity['entity_group'] in ['ORG', 'MISC', 'PER']
                ]
                return relevant_entities
        except (requests.RequestException, ValueError, KeyError, Exception) as e:
            # Log for debugging
            print(f"Error in _classify_intent: {e}")

        return []

    def _basic_parse(self, query: str):
        """Your existing parsing logic as reliable fallback"""
        import re

        academic_stop_words = {
            'paper', 'papers', 'research', 'study', 'studies', 'review',
            'article', 'articles', 'literature', 'survey', 'about', 'on',
            'in', 'the', 'a', 'an', 'and', 'or', 'for', 'of', 'from',
            'with', 'using', 'based', 'approach', 'method', 'methods'
        }

        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = int(year_match.group(1)) if year_match else None

        # Clean the query for keyword extraction
        query_clean = re.sub(r'\b20\d{2}\b', '', query)
        query_clean = re.sub(r'[^\w\s]', ' ', query_clean)

        # Extract keywords
        words = query_clean.lower().split()
        keywords = [word for word in words
                    if word not in academic_stop_words and len(word) > 2]

        return {
            "keywords": keywords,
            "filters": {"year": year} if year else {},
            "original_query": query
        }

    def summarize_abstract(self, abstract: str, max_length: int = 150) -> str:
            """
            Summarize paper abstracts
            """
            url = f"https://api-inference.huggingface.co/models/{self.models['summarization']}"
            headers = {"Authorization": f"Bearer {self.api_token}"}

            payload = {
                "inputs": abstract,
                "parameters": {"max_length": max_length, "min_length": 30}
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    return response.json()[0]['summary_text']
            except (requests.RequestException, ValueError, KeyError, Exception) as e:
                # Log for debugging
                print(f"Error in _classify_intent: {e}")

            # Fallback: return first 150 chars
            return abstract[:147] + "..." if len(abstract) > 150 else abstract


# Singleton instance
huggingface_service = HuggingFaceService()