# search/services/ranking_service.py
from typing import List, Dict
import re


class RelevanceRanker:
    """
    Ranks papers based on relevance to the original query
    Assigns scores from 1 (low relevance) to 5 (high relevance)
    """

    def rank_papers(self, papers: List[Dict], parsed_query: dict) -> List[Dict]:
        """
        Rank papers by relevance and add relevance_score field

        Args:
            papers: List of paper dictionaries
            parsed_query: Parsed query with keywords, entities, intent

        Returns:
            Sorted list of papers with relevance_score added
        """
        # Extract search terms
        keywords = set(parsed_query.get('keywords', []))
        phrases = set(parsed_query.get('phrases', []))
        original_query = parsed_query.get('original_query', '')
        if original_query is None:
            original_query = ''
        original_query = str(original_query).lower()

        # Combine all important terms
        important_terms = keywords | phrases

        # Score each paper
        for paper in papers:
            score = self._calculate_relevance_score(
                paper,
                important_terms,
                original_query,
                parsed_query
            )
            paper['relevance_score'] = score

        # Sort by relevance (highest first)
        ranked_papers = sorted(papers, key=lambda x: x['relevance_score'], reverse=True)

        return ranked_papers

    def _calculate_relevance_score(
            self,
            paper: Dict,
            important_terms: set,
            original_query: str,
            parsed_query: dict
    ) -> int:
        """
        Calculate relevance score (1-5) for a single paper
        """
        score = 0
        max_score = 0

        # Prepare paper text for analysis
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        # Handle None values safely
        if title is None:
            title = ''
        if abstract is None:
            abstract = ''
        title = str(title).lower()
        abstract = str(abstract).lower()
        combined_text = f"{title} {abstract}"

        # 1. Title keyword matches (most important) - up to 30 points
        title_matches = sum(1 for term in important_terms if term.lower() in title)
        score += min(title_matches * 10, 30)
        max_score += 30

        # 2. Abstract keyword matches - up to 25 points
        abstract_matches = sum(1 for term in important_terms if term.lower() in abstract)
        score += min(abstract_matches * 5, 25)
        max_score += 25

        # 3. Exact phrase matching - up to 20 points
        query_phrases = self._extract_phrases(original_query)
        phrase_matches = sum(1 for phrase in query_phrases if phrase in combined_text)
        score += min(phrase_matches * 10, 20)
        max_score += 20

        # 4. Year relevance - up to 15 points
        year_score = self._score_year_relevance(paper, parsed_query)
        score += year_score
        max_score += 15

        # 5. Source bonus - up to 10 points
        source_score = self._score_source_quality(paper)
        score += source_score
        max_score += 10

        # Convert to 1-5 scale
        if max_score == 0:
            return 3  # Default middle score

        percentage = (score / max_score) * 100

        if percentage >= 80:
            return 5  # Highly relevant
        elif percentage >= 60:
            return 4  # Very relevant
        elif percentage >= 40:
            return 3  # Moderately relevant
        elif percentage >= 20:
            return 2  # Somewhat relevant
        else:
            return 1  # Low relevance

    def _extract_phrases(self, query: str) -> List[str]:
        """Extract multi-word phrases from query"""
        if query is None:
            return []
        stop_words = {'in', 'on', 'at', 'the', 'a', 'an', 'for', 'of', 'with', 'from'}
        words = str(query).lower().split()

        phrases = []
        # Look for 2-3 word combinations
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i + 1] not in stop_words:
                phrases.append(f"{words[i]} {words[i + 1]}")

        return phrases

    def _score_year_relevance(self, paper: Dict, parsed_query: dict) -> int:
        """Score based on publication year relevance"""
        from datetime import datetime

        paper_year = paper.get('year')
        if not paper_year:
            return 5  # No penalty if year unknown

        # intent is a string, not a dictionary (e.g., "recent papers", "general search")
        intent_str = str(parsed_query.get('intent', ''))

        filters = parsed_query.get('filters', {})
        year_min = filters.get('year_min')
        year_max = filters.get('year_max')

        # If year filter specified
        if year_min and year_max:
            if year_min <= paper_year <= year_max:
                return 15
            else:
                return 0

        # If "recent" intent, prefer newer papers
        if 'recent' in intent_str.lower() or 'developments' in intent_str.lower():
            current_year = datetime.now().year
            age = current_year - paper_year

            if age <= 1:
                return 15
            elif age <= 2:
                return 12
            elif age <= 3:
                return 8
            elif age <= 5:
                return 5
            else:
                return 2

        # Default: papers from last 5 years get slight bonus
        current_year = datetime.now().year
        if current_year - paper_year <= 5:
            return 10
        else:
            return 5

    def _score_source_quality(self, paper: Dict) -> int:
        """Small bonus for certain sources"""
        source = paper.get('source', '')
        if source is None:
            source = ''
        source = str(source).lower()

        if source in ['pubmed', 'arxiv']:
            return 10

        return 5


# Singleton instance
relevance_ranker = RelevanceRanker()