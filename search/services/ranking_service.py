# search/services/ranking_service.py
from typing import List, Dict
import re
from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


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

        # Get phrases from parsed_query (Gemini already extracted them)
        query_phrases = parsed_query.get('phrases', [])
        query_keywords = parsed_query.get('keywords', [])

        # Break phrases into individual words for flexible matching
        all_search_terms = set()
        for phrase in query_phrases:
            # Add the phrase itself
            all_search_terms.add(phrase.lower())
            # Add individual words from phrase
            all_search_terms.update(phrase.lower().split())
        # Add keywords
        all_search_terms.update(k.lower() for k in query_keywords)

        # 1. Title matching (most important) - up to 50 points
        title_score = 0
        # Exact phrase match in title (highest value)
        exact_phrase_in_title = sum(1 for phrase in query_phrases if phrase.lower() in title)
        title_score += exact_phrase_in_title * 20  # 20 points per exact phrase match
        
        # Individual term matches in title
        title_term_matches = sum(1 for term in all_search_terms if term in title and len(term) > 2)
        title_score += title_term_matches * 5  # 5 points per term match
        
        score += min(title_score, 50)
        max_score += 50

        # 2. Abstract matching - up to 30 points
        abstract_score = 0
        # Exact phrase match in abstract
        exact_phrase_in_abstract = sum(1 for phrase in query_phrases if phrase.lower() in abstract)
        abstract_score += exact_phrase_in_abstract * 10  # 10 points per exact phrase match
        
        # Individual term matches in abstract
        abstract_term_matches = sum(1 for term in all_search_terms if term in abstract and len(term) > 2)
        abstract_score += abstract_term_matches * 2  # 2 points per term match
        
        score += min(abstract_score, 30)
        max_score += 30

        # 3. Year relevance - up to 10 points
        year_score = self._score_year_relevance(paper, parsed_query)
        score += year_score
        max_score += 10

        # 4. Source quality bonus - up to 10 points
        source_score = self._score_source_quality(paper)
        score += source_score
        max_score += 10

        # Convert to 1-5 scale (max_score = 100)
        if max_score == 0:
            return 3  # Default middle score

        percentage = (score / max_score) * 100

        # Debug logging for detailed scoring analysis
        logger.debug(f"Paper: '{title[:60]}...' | Score: {score}/{max_score} ({percentage:.1f}%) | Phrases: {query_phrases} | Stars: {5 if percentage >= 70 else 4 if percentage >= 50 else 3 if percentage >= 30 else 2 if percentage >= 15 else 1}")

        # Adjusted thresholds for better differentiation
        if percentage >= 70:
            return 5  # Highly relevant
        elif percentage >= 50:
            return 4  # Very relevant
        elif percentage >= 30:
            return 3  # Moderately relevant
        elif percentage >= 15:
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
                return 10  # Full points for matching filter
            else:
                return 0  # Zero for outside range

        # If "recent" intent, prefer newer papers
        if 'recent' in intent_str.lower() or 'developments' in intent_str.lower():
            current_year = datetime.now().year
            age = current_year - paper_year

            if age <= 1:
                return 10
            elif age <= 2:
                return 8
            elif age <= 3:
                return 6
            elif age <= 5:
                return 4
            else:
                return 2

        # Default: papers from last 3-5 years get slight bonus
        current_year = datetime.now().year
        if current_year - paper_year <= 3:
            return 7
        elif current_year - paper_year <= 5:
            return 5
        else:
            return 3

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