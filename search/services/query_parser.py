import re


class QueryProcessor:
    def __init__(self):
        self.academic_stop_words = {
            'paper', 'papers', 'research', 'study', 'studies', 'review',
            'article', 'articles', 'literature', 'survey', 'about', 'on',
            'in', 'the', 'a', 'an', 'and', 'or', 'for', 'of', 'from',
            'with', 'using', 'based', 'approach', 'method', 'methods'
        }

    def parse_query(self, query: str) -> dict:
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = int(year_match.group(1)) if year_match else None

        # Clean the query for keyword extraction
        query_clean = re.sub(r'\b20\d{2}\b', '', query)
        query_clean = re.sub(r'[^\w\s]', ' ', query_clean)

        # Extract keywords
        words = query_clean.lower().split()
        keywords = [word for word in words
                    if word not in self.academic_stop_words and len(word) > 2]

        return {
            "keywords": keywords,
            "filters": {"year": year} if year else {},
            "original_query": query
        }


# Create singleton instance
query_processor = QueryProcessor()