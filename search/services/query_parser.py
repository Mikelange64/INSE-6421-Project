# search/services/query_parser.py

class AdvancedQueryProcessor:
    def __init__(self):
        # ✅ Import INSIDE __init__ to avoid circular dependency
        from .llm_service import huggingface_service
        self.llm_service = huggingface_service

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

query_processor = AdvancedQueryProcessor()