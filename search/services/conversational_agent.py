# search/services/conversational_agent.py
import os
import json
import re
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from .api_clients import arxiv_client, pubmed_client
from .ranking_service import relevance_ranker
from .citation_service import citation_formatter

load_dotenv()


class ResearchAgent:
    """
    Conversational AI research assistant using Gemini
    Maintains conversation context and can have back-and-forth
    """

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash (available in free tier)
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except:
            # Fallback to alternative model names if that fails
            try:
                self.model = genai.GenerativeModel('gemini-pro')
            except:
                self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Conversation history (per session)
        self.conversations = {}

        # Current search results (per session)
        self.search_results = {}

        # Last query context (per session)
        self.query_context = {}

    def get_or_create_chat(self, session_id: str):
        """Get existing chat session or create new one"""
        if session_id not in self.conversations:
            self.conversations[session_id] = self.model.start_chat(history=[])
        return self.conversations[session_id]

    def chat(self, user_message: str, session_id: str = "default") -> dict:
        """
        Main conversation handler with TWO-STEP process:
        1. Detect if this is a search request
        2. If yes, extract structured query using few-shot prompting
        """
        try:
            # STEP 1: Detect intent (simple, reliable)
            is_search = self._detect_search_intent(user_message)

            print(f"Intent: {'SEARCH' if is_search else 'CONVERSATION'}")

            if is_search:
                # STEP 2: Extract structured query using few-shot prompting
                parsed_query = self._extract_query_parameters(user_message)

                if parsed_query:
                    print(f"Parsed query: {parsed_query}")

                    # STEP 3: Validate query (check for ambiguity/issues)
                    validation = self._validate_query(parsed_query, user_message)

                    if validation['needs_clarification']:
                        print(f"Query needs clarification: {validation['reason']}")
                        return {
                            'agent_response': validation['message'],
                            'action': 'clarification',
                            'data': {
                                'reason': validation['reason'],
                                'suggestions': validation.get('suggestions', [])
                            }
                        }

                    # Execute search
                    results = self._execute_search(parsed_query)
                    self.search_results[session_id] = results
                    self.query_context[session_id] = parsed_query

                    # Generate friendly response
                    result_message = self._generate_result_message(results, user_message)

                    return {
                        'agent_response': result_message,
                        'action': 'search',
                        'data': {
                            'results': results,
                            'count': len(results)
                        }
                    }
                else:
                    # Parsing failed, ask for clarification
                    return {
                        'agent_response': "I couldn't understand your search query. Could you rephrase it? For example: 'papers on machine learning from 2020' or 'recent quantum computing research'",
                        'action': 'error',
                        'data': {}
                    }
            else:
                # Regular conversation
                try:
                    chat = self.get_or_create_chat(session_id)
                    response = chat.send_message(self._build_conversational_prompt(user_message))

                    return {
                        'agent_response': response.text,
                        'action': 'conversation',
                        'data': {}
                    }
                except Exception as conv_error:
                    print(f"Conversation error: {conv_error}")
                    # Graceful fallback: guide user to search instead
                    return {
                        'agent_response': "I'm currently focused on helping you search for academic papers. What research topic would you like to explore? For example, try asking 'Find papers on machine learning' or 'Recent research on climate change'.",
                        'action': 'conversation',
                        'data': {}
                    }

        except Exception as e:
            print(f"Agent error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'agent_response': "I encountered an error. Could you rephrase your request?",
                'action': 'error',
                'data': {'error': str(e)}
            }

    def _detect_search_intent(self, user_message: str) -> bool:
        """
        Step 1: Enhanced intent detection with contextual understanding
        Returns True if user wants to search for papers
        """
        message_lower = user_message.lower()

        # CLEAR SEARCH INTENT - Direct requests for papers/research
        search_keywords = [
            'paper', 'papers', 'research', 'find', 'search', 'article', 'articles',
            'study', 'studies', 'publication', 'publications', 'arxiv', 'pubmed',
            'literature', 'bibliography', 'references', 'citations', 'sources'
        ]
        if any(keyword in message_lower for keyword in search_keywords):
            return True

        # CLEAR CONVERSATIONAL INTENT - Direct conversation starters
        conversational_phrases = [
            'hello', 'hi', 'hey', 'greetings', 'thanks', 'thank you',
            'what can you do', 'how are you', 'who are you', 'help',
            'good morning', 'good afternoon', 'good evening',
            'nice to meet you', 'how do you work', 'tell me about yourself'
        ]
        if any(phrase in message_lower for phrase in conversational_phrases):
            return False

        # CONTEXTUAL SEARCH INTENT - Academic/research activities
        academic_activities = [
            'writing a research', 'working on a paper', 'doing research on',
            'researching', 'studying', 'analyzing', 'investigating',
            'literature review', 'thesis', 'dissertation', 'dissertation on',
            'phd research', 'masters thesis', 'academic paper',
            'scholarly article', 'journal article', 'conference paper',
            'peer reviewed', 'systematic review', 'meta analysis', 'doing a homework assignment',
        ]
        if any(activity in message_lower for activity in academic_activities):
            return True

        # INFORMATION SEEKING - Looking for knowledge
        info_seeking = [
            'looking for information', 'need information on', 'trying to find',
            'seeking sources on', 'interested in', 'curious about',
            'want to learn about', 'need to know about', 'tell me about',
            'explain', 'what do you know about', 'give me information on'
        ]
        if any(seeking in message_lower for seeking in info_seeking):
            return True

        # TOPIC EXPLORATION - Questions about specific topics
        topic_indicators = [
            'what is', 'how does', 'why does', 'when did', 'where did',
            'how do', 'what are the', 'what was', 'can you explain',
            'tell me more about', 'i want to understand', 'help me understand'
        ]
        if any(indicator in message_lower for indicator in topic_indicators):
            return True

        # AMBIGUOUS CASES - Use Gemini for nuanced understanding
        try:
            prompt = f"""Determine if this message is asking to search for academic papers or research information.

Look for:
- Research activities (writing papers, literature reviews, etc.)
- Information seeking about academic topics
- Questions that would benefit from scholarly sources
- Academic or research contexts

Examples of SEARCH intent:
- "I'm writing a paper on climate change"
- "Need sources for my thesis on quantum physics"
- "Find research on machine learning applications"
- "What are the latest papers on COVID-19 vaccines?"

Examples of CONVERSATION intent:
- "Hello, how are you?"
- "What can you help me with?"
- "Tell me about your features"

User message: "{user_message}"

Answer with ONLY: SEARCH or CONVERSATION"""

            response = self.model.generate_content(prompt)
            answer = response.text.strip().upper()

            return 'SEARCH' in answer

        except Exception as e:
            print(f"Intent detection error: {e}")
            # For ambiguous cases, default to search since this is an academic research tool
            return True

    def _extract_query_parameters(self, user_message: str) -> dict:
        """
        Step 2: Extract structured query parameters using FEW-SHOT prompting
        Returns parsed_query dict or None if parsing fails
        """
        current_year = datetime.now().year

        # FEW-SHOT PROMPT with concrete examples
        prompt = f"""Extract search parameters from the query. Output ONLY valid JSON, no other text.

EXAMPLES:

Input: "papers on machine learning from past 5 years"
Output: {{"phrases": ["machine learning"], "keywords": [], "filters": {{"year_min": {current_year - 5}, "year_max": {current_year}}}, "intent": "recent papers"}}

Input: "Haitian Revolution impact"
Output: {{"phrases": ["Haitian Revolution"], "keywords": ["impact"], "filters": {{}}, "intent": "general search"}}

Input: "recent quantum computing research"
Output: {{"phrases": ["quantum computing"], "keywords": ["research"], "filters": {{"year_min": {current_year - 3}, "year_max": {current_year}}}, "intent": "recent research"}}

Input: "AI ethics papers from {current_year - 5} to {current_year - 1}"
Output: {{"phrases": ["AI ethics"], "keywords": [], "filters": {{"year_min": {current_year - 5}, "year_max": {current_year - 1}}}, "intent": "specific time range"}}

Input: "neural networks in healthcare"
Output: {{"phrases": ["neural networks"], "keywords": ["healthcare"], "filters": {{}}, "intent": "general search"}}

RULES:
- Multi-word technical terms go in "phrases" (e.g., "machine learning", "Haitian Revolution")
- Single words go in "keywords"
- "past X years" means year_min = {current_year} - X, year_max = {current_year}
- "recent" means year_min = {current_year - 3}, year_max = {current_year}
- Current year is {current_year}

Now parse this query:
Input: "{user_message}"
Output:"""

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            print(f"Raw Gemini response:\n{raw_text}\n")

            # Try to parse JSON
            parsed = self._parse_json_response(raw_text)

            if parsed:
                # CRITICAL: Always create all_terms
                parsed['all_terms'] = parsed.get('phrases', []) + parsed.get('keywords', [])
                parsed['original_query'] = user_message

                # Ensure required fields exist
                if 'phrases' not in parsed:
                    parsed['phrases'] = []
                if 'keywords' not in parsed:
                    parsed['keywords'] = []
                if 'filters' not in parsed:
                    parsed['filters'] = {}
                if 'intent' not in parsed:
                    parsed['intent'] = 'general search'

                return parsed
            else:
                # JSON parsing failed, try regex fallback
                print("JSON parsing failed, trying fallback extraction")
                return self._fallback_query_extraction(user_message)

        except Exception as e:
            print(f"Query extraction error: {e}")
            return self._fallback_query_extraction(user_message)

    def _parse_json_response(self, text: str) -> dict:
        """
        Extract JSON from Gemini response
        Handles markdown code blocks and extra text
        """
        try:
            # Remove markdown code blocks
            if '```json' in text:
                start = text.find('```json') + 7
                end = text.find('```', start)
                text = text[start:end].strip()
            elif '```' in text:
                start = text.find('```') + 3
                end = text.find('```', start)
                text = text[start:end].strip()

            # Find JSON object
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]

                parsed = json.loads(json_str)
                return parsed

            return None

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def _fallback_query_extraction(self, user_message: str) -> dict:
        """
        Fallback: Rule-based query extraction when AI fails
        Uses regex and simple heuristics
        """
        current_year = datetime.now().year

        # Extract year ranges using regex
        filters = {}

        # Pattern: "from 2020 to 2023"
        year_range_match = re.search(r'from\s+(\d{4})\s+to\s+(\d{4})', user_message, re.IGNORECASE)
        if year_range_match:
            filters['year_min'] = int(year_range_match.group(1))
            filters['year_max'] = int(year_range_match.group(2))

        # Pattern: "past X years"
        past_years_match = re.search(r'past\s+(\d+)\s+years?', user_message, re.IGNORECASE)
        if past_years_match:
            num_years = int(past_years_match.group(1))
            filters['year_min'] = current_year - num_years
            filters['year_max'] = current_year

        # Pattern: "recent" (assume 3 years)
        if 'recent' in user_message.lower():
            filters['year_min'] = current_year - 3
            filters['year_max'] = current_year

        # Pattern: single year "2023"
        single_year_match = re.search(r'\b(19|20)\d{2}\b', user_message)
        if single_year_match and not filters:
            year = int(single_year_match.group(0))
            filters['year_min'] = year
            filters['year_max'] = year

        # Extract keywords (simple: split by spaces, remove common words)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'from', 'to', 'for', 'of', 'with',
                      'papers', 'paper', 'research', 'find', 'search', 'about', 'recent',
                      'past', 'years', 'year', 'writing', "i'm", "i'll", "i've", "i'd",
                      'need', 'needs', 'sources', 'source', 'thesis', 'theses', 'dissertation',
                      'writing', 'write', 'working', 'work', 'doing', 'do', 'making', 'make',
                      'creating', 'create', 'looking', 'look', 'seeking', 'seek', 'want',
                      'wants', 'wanted', 'trying', 'try', 'going', 'goes', 'go'}

        words = user_message.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Try to detect phrases (capitalized consecutive words)
        phrases = []
        phrase_match = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', user_message)
        if phrase_match:
            phrases = phrase_match

        # Build parsed query
        parsed = {
            'phrases': phrases,
            'keywords': keywords,
            'all_terms': phrases + keywords,
            'filters': filters,
            'intent': 'general search',
            'original_query': user_message
        }

        print(f"Fallback extraction result: {parsed}")
        return parsed

    def _validate_query(self, parsed_query: dict, original_message: str) -> dict:
        """
        Validate parsed query and check for issues that need clarification

        Returns:
            dict with keys: needs_clarification (bool), reason (str), message (str), suggestions (list)
        """
        all_terms = parsed_query.get('all_terms', [])

        # Check for empty query
        if not all_terms or all(term.strip() == '' for term in all_terms):
            return {
                'needs_clarification': True,
                'reason': 'empty_query',
                'message': "I couldn't find any search terms in your query. Could you please specify what you're looking for? For example: 'papers on machine learning' or 'recent research on climate change'",
                'suggestions': []
            }

        # Check for political terms
        political_terms = [
            'left-wing', 'right-wing', 'liberal', 'conservative', 'democrat', 'republican',
            'political', 'politics', 'election', 'government', 'policy', 'ideology',
            'socialism', 'communism', 'capitalism', 'fascism', 'authoritarian'
        ]

        query_text = ' '.join(all_terms).lower()
        if any(term in query_text for term in political_terms):
            return {
                'needs_clarification': True,
                'reason': 'political_terms',
                'message': "I notice your query contains political terms. For academic research, I recommend focusing on specific research topics rather than political viewpoints. Could you rephrase your query to focus on the academic subject matter?",
                'suggestions': [
                    "Try: 'research on economic inequality'",
                    "Try: 'studies on democratic institutions'",
                    "Try: 'academic analysis of policy effectiveness'"
                ]
            }

        # Check for subjective terms
        subjective_terms = [
            'best', 'worst', 'top', 'greatest', 'most important', 'leading',
            'excellent', 'outstanding', 'superior', 'inferior', 'controversial',
            'debunked', 'proven', 'disproven', 'fake', 'real'
        ]

        if any(term in query_text for term in subjective_terms):
            return {
                'needs_clarification': True,
                'reason': 'subjective_terms',
                'message': "Your query contains subjective terms that might not yield reliable academic results. Academic search works best with specific topics and methodologies. Could you rephrase to focus on objective research areas?",
                'suggestions': [
                    "Instead of 'best AI papers', try 'recent advances in artificial intelligence'",
                    "Instead of 'most important climate research', try 'climate change mitigation strategies'"
                ]
            }

        # Check for very short terms that might be typos
        if len(all_terms) == 1 and len(all_terms[0]) <= 2:
            term = all_terms[0]
            if term.lower() not in ['ai', 'ml', 'cs', 'it', 'vr', 'ar', 'io', 'ui', 'ux', 'os']:
                return {
                    'needs_clarification': True,
                    'reason': 'too_short',
                    'message': f"'{term}' is very short and might not give good results. Did you mean something more specific?",
                    'suggestions': [
                        f"Try: '{term} research' or '{term} studies'",
                        f"Try: 'recent {term} developments'",
                        f"Try: '{term} applications'"
                    ]
                }

        # Query is valid
        return {
            'needs_clarification': False,
            'reason': None,
            'message': None,
            'suggestions': []
        }

    def _build_conversational_prompt(self, user_message: str) -> str:
        """
        Build prompt for conversational responses (non-search)
        """
        return f"""You are Mayele, a helpful AI research assistant.

The user is having a conversation with you (NOT searching for papers right now).

Respond naturally and helpfully. Keep responses concise.

User: {user_message}"""

    def _execute_search(self, parsed_query: dict) -> list:
        """
        Execute search using parsed query
        """
        print(f"Searching with query: {parsed_query}")

        # Search both APIs
        arxiv_results = arxiv_client.search(parsed_query)
        pubmed_results = pubmed_client.search(parsed_query)

        all_results = arxiv_results + pubmed_results

        # Rank by relevance
        if all_results:
            all_results = relevance_ranker.rank_papers(all_results, parsed_query)

        print(f"Found {len(all_results)} papers")
        return all_results

    def _generate_result_message(self, results: list, original_query: str) -> str:
        """
        Generate a friendly message about search results
        """
        count = len(results)

        if count == 0:
            return f"I couldn't find any papers matching '{original_query}'. Try rephrasing or broadening your search."
        elif count == 1:
            return f"I found 1 paper matching your query. Check it out below!"
        elif count <= 5:
            return f"I found {count} papers matching your query. Here they are:"
        else:
            return f"I found {count} papers matching your query. Here are the most relevant ones:"

    def get_citations(self, session_id: str, paper_indices: list, style: str = 'apa') -> list:
        """
        Get citations for specific papers from last search
        """
        if session_id not in self.search_results:
            return []

        results = self.search_results[session_id]
        citations = []

        for idx in paper_indices:
            if 0 <= idx < len(results):
                paper = results[idx]
                citation = citation_formatter.format_citation(paper, style)
                citations.append(citation)

        return citations

    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.search_results:
            del self.search_results[session_id]
        if session_id in self.query_context:
            del self.query_context[session_id]


# Singleton instance
research_agent = ResearchAgent()