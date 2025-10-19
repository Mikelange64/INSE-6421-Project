# search/tests.py
from django.test import TestCase
from unittest.mock import patch, MagicMock
from .services.llm_service import HuggingFaceService
from .services.query_parser import AdvancedQueryProcessor


class LLMServiceTests(TestCase):
    def setUp(self):
        self.service = HuggingFaceService()

    def test_basic_parse_extracts_year(self):
        """Test that year extraction works"""
        result = self.service._basic_parse("AI research from 2023")
        self.assertEqual(result['filters']['year'], 2023)

    def test_basic_parse_extracts_keywords(self):
        """Test keyword extraction filters stop words"""
        result = self.service._basic_parse("machine learning in healthcare")
        self.assertIn('machine', result['keywords'])
        self.assertIn('learning', result['keywords'])
        self.assertIn('healthcare', result['keywords'])
        # Stop words should be filtered
        self.assertNotIn('in', result['keywords'])

    @patch('search.services.llm_service.requests.post')
    def test_classify_intent_success(self, mock_post):
        """Test intent classification with mocked API response"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'labels': ['find specific papers', 'literature review'],
            'scores': [0.85, 0.15]
        }
        mock_post.return_value = mock_response

        result = self.service._classify_intent("Find papers on AI")

        self.assertEqual(result['primary_intent'], 'find specific papers')
        self.assertEqual(result['confidence'], 0.85)

    @patch('search.services.llm_service.requests.post')
    def test_classify_intent_fallback_on_error(self, mock_post):
        """Test that fallback works when API fails"""
        # Mock API failure
        mock_post.side_effect = Exception("API Error")

        result = self.service._classify_intent("Find papers on AI")

        # Should return fallback values
        self.assertEqual(result['primary_intent'], 'find specific papers')
        self.assertEqual(result['confidence'], 0.8)


class QueryProcessorTests(TestCase):
    def setUp(self):
        self.processor = AdvancedQueryProcessor()

    @patch.object(HuggingFaceService, 'analyze_research_query')
    def test_parse_query_combines_analysis(self, mock_analyze):
        """Test that parse_query combines HF and basic analysis"""
        # Mock HF service response
        mock_analyze.return_value = {
            'intent': {'primary_intent': 'find specific papers'},
            'entities': []
        }

        result = self.processor.parse_query("machine learning 2023")

        # Should have both HF analysis and basic parsing
        self.assertIn('intent', result)
        self.assertIn('keywords', result)
        self.assertEqual(result['original_query'], "machine learning 2023")


class SearchViewTests(TestCase):
    def test_home_view_loads(self):
        """Test that home page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_search_api_requires_query(self):
        """Test that search API returns error for empty query"""
        response = self.client.get('/api/search/')
        self.assertEqual(response.status_code, 400)

    @patch('search.services.api_clients.arxiv_client.search')
    @patch('search.services.api_clients.pubmed_client.search')
    def test_search_api_with_valid_query(self, mock_pubmed, mock_arxiv):
        """Test search API with mocked external services"""
        # Mock API responses
        mock_arxiv.return_value = [{'title': 'Test Paper', 'source': 'arxiv'}]
        mock_pubmed.return_value = []

        response = self.client.get('/api/search/?q=machine learning')

        self.assertEqual(response.status_code, 200)
        # Check that both APIs were called
        mock_arxiv.assert_called_once()
        mock_pubmed.assert_called_once()