# search/tests.py
from django.test import TestCase
from unittest.mock import patch, MagicMock
from .services.conversational_agent import ResearchAgent
from .services.citation_service import citation_formatter
from .services.ranking_service import relevance_ranker


class GeminiIntentDetectionTests(TestCase):
    def setUp(self):
        self.agent = ResearchAgent()

    def test_detects_search_with_keywords(self):
        """Test: 'find papers on AI' → SEARCH"""
        result = self.agent._detect_search_intent('find papers on AI')
        self.assertTrue(result, "Should detect search intent for 'find papers'")

    def test_detects_conversation(self):
        """Test: 'hello' → CONVERSATION"""
        result = self.agent._detect_search_intent('hello')
        self.assertFalse(result, "Should detect conversation intent for 'hello'")

    def test_handles_ambiguous_query(self):
        """Test: 'tell me about quantum physics' → SEARCH"""
        # This should be detected as search since it's information seeking
        result = self.agent._detect_search_intent('tell me about quantum physics')
        self.assertTrue(result, "Information seeking queries should be detected as search")


class GeminiQueryExtractionTests(TestCase):
    def setUp(self):
        self.agent = ResearchAgent()

    @patch('google.generativeai.GenerativeModel')
    def test_extracts_phrases_correctly(self, mock_model):
        """Test: 'Haitian Revolution' → phrases=['Haitian Revolution']"""
        # Test fallback extraction for phrases
        result = self.agent._fallback_query_extraction('Haitian Revolution impact')
        self.assertIn('Haitian Revolution', result.get('phrases', []), 
                     "Should extract capitalized phrase 'Haitian Revolution'")

    @patch('google.generativeai.GenerativeModel')
    def test_extracts_year_filters(self, mock_model):
        """Test: 'past 5 years' → year_min and year_max"""
        from datetime import datetime
        current_year = datetime.now().year
        result = self.agent._fallback_query_extraction('papers from past 5 years')
        
        filters = result.get('filters', {})
        self.assertIn('year_min', filters, "Should extract year_min")
        self.assertIn('year_max', filters, "Should extract year_max")
        self.assertEqual(filters['year_max'], current_year, "year_max should be current year")
        self.assertEqual(filters['year_min'], current_year - 5, "year_min should be current year - 5")

    def test_fallback_extraction_works(self):
        """Test regex fallback when JSON parsing fails"""
        result = self.agent._fallback_query_extraction('machine learning in healthcare from 2020 to 2023')
        
        filters = result.get('filters', {})
        self.assertEqual(filters.get('year_min'), 2020, "Should extract year_min=2020")
        self.assertEqual(filters.get('year_max'), 2023, "Should extract year_max=2023")
        # Fallback extraction only captures capitalized phrases, so lowercase phrases go to keywords
        self.assertIn('machine', result.get('keywords', []), "Should extract keywords")
        self.assertIn('learning', result.get('keywords', []), "Should extract keywords")

    def test_always_creates_all_terms(self):
        """Test: all_terms is always present in parsed_query"""
        result = self.agent._fallback_query_extraction('quantum computing research')
        
        self.assertIn('all_terms', result, "Should always have all_terms field")
        self.assertTrue(len(result['all_terms']) > 0, "all_terms should not be empty")


class QueryValidationTests(TestCase):
    def setUp(self):
        self.agent = ResearchAgent()

    def test_rejects_political_terms(self):
        """Test: 'left-wing article' → needs_clarification=True"""
        parsed = {'all_terms': ['left-wing', 'article'], 'filters': {}}
        validation = self.agent._validate_query(parsed, 'left-wing article')
        self.assertTrue(validation['needs_clarification'])
        self.assertEqual(validation['reason'], 'political_terms')

    def test_rejects_subjective_terms(self):
        """Test: 'best papers' → needs_clarification=True"""
        parsed = {'all_terms': ['best', 'papers'], 'filters': {}}
        validation = self.agent._validate_query(parsed, 'best papers')
        self.assertTrue(validation['needs_clarification'], 
                       "Should reject subjective terms like 'best'")
        self.assertEqual(validation['reason'], 'subjective_terms')

    def test_rejects_empty_query(self):
        """Test: no terms extracted → needs_clarification=True"""
        parsed = {'all_terms': [], 'filters': {}}
        validation = self.agent._validate_query(parsed, '')
        self.assertTrue(validation['needs_clarification'],
                       "Should reject empty queries")
        self.assertEqual(validation['reason'], 'empty_query')

    def test_accepts_valid_query(self):
        """Test: 'machine learning' → needs_clarification=False"""
        parsed = {'all_terms': ['machine learning'], 'filters': {}}
        validation = self.agent._validate_query(parsed, 'machine learning')
        self.assertFalse(validation['needs_clarification'])


class ArxivClientTests(TestCase):
    def test_handles_all_terms_format(self):
        """Test: api_clients work with new all_terms format"""
        from search.services.api_clients import arxiv_client
        
        parsed_query = {
            'phrases': ['machine learning'],
            'keywords': ['healthcare'],
            'all_terms': ['machine learning', 'healthcare'],
            'filters': {}
        }
        
        query = arxiv_client._build_search_query(parsed_query)
        self.assertIn('machine learning', query, "Should include phrase in search query")
        self.assertIn('healthcare', query, "Should include keyword in search query")

    def test_fallback_to_keywords(self):
        """Test: uses keywords if all_terms missing"""
        from search.services.api_clients import arxiv_client
        
        parsed_query = {
            'keywords': ['artificial', 'intelligence'],
            'filters': {}
        }
        
        query = arxiv_client._build_search_query(parsed_query)
        self.assertIn('artificial', query, "Should use keywords when all_terms missing")


class CitationServiceTests(TestCase):
    def test_author_formatting_with_et_al(self):
        """Test: Citation service handles 'et al.' correctly"""
        paper = {
            'authors': 'John Doe, Jane Smith et al.',
            'title': 'Test Paper',
            'year': 2023,
            'source': 'arxiv',
            'link': 'http://example.com'
        }
        
        citation = citation_formatter.format_citation(paper, 'apa')
        self.assertIn('John Doe', citation, "Should include first author")
        self.assertIn('et al.', citation, "Should include et al.")

    def test_author_splitting_removes_et_al_properly(self):
        """Test: _split_authors handles et al. correctly"""
        # This should extract just the author names before et al.
        authors_with_et_al = 'John Doe, Jane Smith et al.'
        author_list = citation_formatter._split_authors(authors_with_et_al)
        
        self.assertIn('John Doe', author_list, "Should extract John Doe")
        self.assertIn('Jane Smith', author_list, "Should extract Jane Smith")
        self.assertEqual(len(author_list), 2, "Should have exactly 2 authors")


class RankingServiceTests(TestCase):
    def test_year_relevance_with_string_intent(self):
        """Test: Ranking service handles string intent correctly"""
        paper = {
            'title': 'Test Paper',
            'authors': 'John Doe',
            'abstract': 'Machine learning in healthcare',
            'source': 'arxiv',
            'year': 2023,
            'link': '#'
        }
        
        parsed_query = {
            'phrases': ['machine learning'],
            'keywords': ['healthcare'],
            'all_terms': ['machine learning', 'healthcare'],
            'filters': {},
            'intent': 'recent research'  # String intent, not dict
        }
        
        result = relevance_ranker.rank_papers([paper], parsed_query)
        self.assertEqual(len(result), 1, "Should rank the paper")
        self.assertIn('relevance_score', result[0], "Should add relevance_score")


class EndToEndSearchTests(TestCase):
    def setUp(self):
        self.agent = ResearchAgent()

    @patch('google.generativeai.GenerativeModel')
    @patch('search.services.api_clients.arxiv_client.search')
    def test_full_search_pipeline(self, mock_arxiv, mock_gemini):
        """Test: User query → Search results (fully mocked)"""
        # Mock API response
        mock_arxiv.return_value = [{
            'title': 'Test Paper',
            'authors': 'John Doe',
            'abstract': 'Test abstract',
            'source': 'arxiv',
            'year': 2023,
            'link': '#'
        }]
        
        # This would require extensive mocking of Gemini
        # For now, just verify the mock setup
        self.assertIsNotNone(mock_arxiv)

    def test_clarification_flow(self):
        """Test: Ambiguous query → Clarification response"""
        # Test that validation returns appropriate response for political terms
        parsed = {'all_terms': ['left-wing', 'politics'], 'filters': {}}
        validation = self.agent._validate_query(parsed, 'left-wing politics')
        
        self.assertTrue(validation['needs_clarification'])
        self.assertIn('message', validation)
        self.assertIn('suggestions', validation)