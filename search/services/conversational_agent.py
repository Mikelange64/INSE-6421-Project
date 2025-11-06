# search/services/conversational_agent.py
import os
import json
import re
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from .api_clients import arxiv_client, pubmed_client, semantic_scholar_client
from .ranking_service import relevance_ranker
from .citation_service import citation_formatter
from .logging_config import get_logger

load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Valid 2-letter academic acronyms/abbreviations (lowercase)
VALID_SHORT_TERMS = {
    # AI/CS/Tech
    'ai', 'ml', 'dl', 'rl', 'nn', 'cv', 'nlp', 'it', 'ar', 'vr', 'ui', 'ux', 'os', 'db', 'ip', 
    # Science/Physics/Chemistry
    'iq', 'eq', 'uv', 'ir', 'ph', 'em', 'gc', 'ms', 'nm', 'hz', 'mw', 'rf',
    # Medicine/Biology
    'ct', 'mr', 'bp', 'tb', 'hiv', 'dna', 'rna', 'icu', 'ed', 'or',
    # Geography/Organizations
    'us', 'uk', 'eu', 'un', 'who',
}


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
            logger.info(f"User query received: '{user_message[:100]}...'")
            
            # STEP 1: Detect intent (simple, reliable)
            is_search = self._detect_search_intent(user_message)
            intent_type = 'SEARCH' if is_search else 'CONVERSATION'
            logger.debug(f"Intent detected: {intent_type}")

            if is_search:
                # STEP 2: Extract structured query using few-shot prompting
                parsed_query = self._extract_query_parameters(user_message)

                if parsed_query:
                    logger.debug(f"Query parsed successfully: phrases={parsed_query.get('phrases', [])}, keywords={parsed_query.get('keywords', [])}, filters={parsed_query.get('filters', {})}")

                    # STEP 3: Validate query (check for ambiguity/issues)
                    validation = self._validate_query(parsed_query, user_message)

                    if validation['needs_clarification']:
                        logger.info(f"Query needs clarification: {validation['reason']}")
                        return {
                            'agent_response': validation['message'],
                            'action': 'clarification',
                            'data': {
                                'reason': validation['reason'],
                                'suggestions': validation.get('suggestions', [])
                            }
                        }

                    # Execute search
                    logger.info("Initiating search across arXiv and PubMed...")
                    results = self._execute_search(parsed_query)
                    self.search_results[session_id] = results
                    self.query_context[session_id] = parsed_query

                    logger.info(f"Search completed: {len(results)} papers found")

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
                    logger.warning("Query parsing failed, requesting clarification from user")
                    return {
                        'agent_response': "I couldn't understand your search query. Could you rephrase it? For example: 'papers on machine learning from 2020' or 'recent quantum computing research'",
                        'action': 'error',
                        'data': {}
                    }
            else:
                # Regular conversation
                try:
                    logger.debug("Handling conversational query")
                    chat = self.get_or_create_chat(session_id)
                    response = chat.send_message(self._build_conversational_prompt(user_message))

                    logger.debug("Conversational response generated successfully")
                    return {
                        'agent_response': response.text,
                        'action': 'conversation',
                        'data': {}
                    }
                except Exception as conv_error:
                    logger.warning(f"Conversation error, using fallback response: {conv_error}")
                    # Graceful fallback: guide user to search instead
                    return {
                        'agent_response': "I'm currently focused on helping you search for academic papers. What research topic would you like to explore? For example, try asking 'Find papers on machine learning' or 'Recent research on climate change'.",
                        'action': 'conversation',
                        'data': {}
                    }

        except Exception as e:
            logger.error(f"Critical error in chat handler: {e}", exc_info=True)
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

        # CLEAR CONVERSATIONAL INTENT - Direct conversation starters (check after search intent)
        conversational_phrases = [
            'hello', 'hi', 'hey', 'greetings', 'thanks', 'thank you',
            'what can you do', 'how are you', 'who are you', 'help',
            'good morning', 'good afternoon', 'good evening',
            'nice to meet you', 'how do you work', 'tell me about yourself'
        ]
        if any(phrase in message_lower for phrase in conversational_phrases):
            return False

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

        # FEW-SHOT PROMPT with concrete examples - Enhanced for better extraction
        prompt = f"""You are extracting search parameters for an academic paper search engine. Extract ONLY the core research topic, ignoring conversational fluff.

Output ONLY valid JSON, no other text.

EXAMPLES:

Input: "papers on machine learning from past 5 years"
Output: {{"phrases": ["machine learning"], "keywords": [], "filters": {{"year_min": {current_year - 5}, "year_max": {current_year}}}, "intent": "recent papers"}}

Input: "Haitian Revolution impact"
Output: {{"phrases": ["Haitian Revolution"], "keywords": ["impact"], "filters": {{}}, "intent": "general search"}}

Input: "I'm writing a research paper on climate change"
Output: {{"phrases": ["climate change"], "keywords": [], "filters": {{}}, "intent": "general search"}}

Input: "recent quantum computing research"
Output: {{"phrases": ["quantum computing"], "keywords": [], "filters": {{"year_min": {current_year - 3}, "year_max": {current_year}}}, "intent": "recent research"}}

Input: "AI ethics papers from {current_year - 5} to {current_year - 1}"
Output: {{"phrases": ["AI ethics"], "keywords": [], "filters": {{"year_min": {current_year - 5}, "year_max": {current_year - 1}}}, "intent": "specific time range"}}

Input: "neural networks in healthcare"
Output: {{"phrases": ["neural networks"], "keywords": ["healthcare"], "filters": {{}}, "intent": "general search"}}

Input: "I need sources for my thesis on artificial intelligence"
Output: {{"phrases": ["artificial intelligence"], "keywords": [], "filters": {{}}, "intent": "general search"}}

Input: "machine learning papers from 2020 to 2023"
Output: {{"phrases": ["machine learning"], "keywords": [], "filters": {{"year_min": 2020, "year_max": 2023}}, "intent": "specific time range"}}

RULES:
1. Extract ONLY the research topic, ignore words like: "I'm writing", "I need", "find me", "papers on", "research on", "sources for"
2. Multi-word technical/academic terms MUST go in "phrases" (e.g., "machine learning", "climate change", "Haitian Revolution", "quantum computing")
3. Only use "keywords" for single descriptive words that modify the topic (e.g., "impact", "applications", "methods")
4. Prioritize extracting 1-3 core phrases over many keywords
5. Time filters:
   - "past X years" → year_min = {current_year} - X, year_max = {current_year}
   - "recent" → year_min = {current_year - 3}, year_max = {current_year}
   - Explicit years → use those exact years
6. Current year is {current_year}

Now parse this query:
Input: "{user_message}"
Output:"""

        try:
            logger.debug("Calling Gemini API for query parameter extraction...")
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            logger.debug(f"Raw Gemini response: {raw_text[:200]}...")

            # Try to parse JSON
            parsed = self._parse_json_response(raw_text)

            if parsed:
                # Ensure required fields exist
                if 'phrases' not in parsed:
                    parsed['phrases'] = []
                if 'keywords' not in parsed:
                    parsed['keywords'] = []
                if 'filters' not in parsed:
                    parsed['filters'] = {}
                if 'intent' not in parsed:
                    parsed['intent'] = 'general search'

                # Post-process: Clean and prioritize terms
                parsed = self._clean_extracted_terms(parsed)

                # CRITICAL: Always create all_terms after cleaning
                parsed['all_terms'] = parsed.get('phrases', []) + parsed.get('keywords', [])
                parsed['original_query'] = user_message

                logger.debug(f"Gemini extraction successful: {parsed}")
                return parsed
            else:
                # JSON parsing failed, try regex fallback
                logger.warning("Gemini JSON parsing failed, using regex fallback")
                return self._fallback_query_extraction(user_message)

        except Exception as e:
            logger.warning(f"Gemini query extraction error: {e}, using fallback")
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

        # Extract keywords (enhanced: split by spaces, remove common words, prioritize domain terms)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'from', 'to', 'for', 'of', 'with',
                      'papers', 'paper', 'research', 'find', 'search', 'about', 'recent',
                      'past', 'years', 'year', 'writing', "i'm", "i'll", "i've", "i'd",
                      'need', 'needs', 'sources', 'source', 'thesis', 'theses', 'dissertation',
                      'writing', 'write', 'working', 'work', 'doing', 'do', 'making', 'make',
                      'creating', 'create', 'looking', 'look', 'seeking', 'seek', 'want',
                      'wants', 'wanted', 'trying', 'try', 'going', 'goes', 'go',
                      'would', 'could', 'should', 'might', 'will', 'can', 'may',
                      'some', 'many', 'much', 'good', 'best', 'better', 'worst', 'great',
                      'new', 'old', 'first', 'last', 'next', 'previous', 'current',
                      'how', 'what', 'why', 'when', 'where', 'which', 'who',
                      'this', 'that', 'these', 'those', 'there', 'here', 'very',
                      'just', 'only', 'also', 'even', 'still', 'already', 'yet'}

        words = user_message.lower().split()
        # Filter out stop words and very short words
        keywords = [word for word in words if word not in stop_words and (len(word) > 2 or word in VALID_SHORT_TERMS)]

        # Prioritize technical/academic domain terms
        domain_terms = {'machine', 'learning', 'artificial', 'intelligence', 'neural', 'network',
                       'deep', 'quantum', 'computing', 'algorithm', 'data', 'science', 'computer',
                       'vision', 'nlp', 'natural', 'language', 'processing', 'reinforcement',
                       'supervised', 'unsupervised', 'classification', 'regression', 'clustering',
                       'optimization', 'statistics', 'probability', 'bayesian', 'graph',
                       'distributed', 'parallel', 'convolutional', 'recurrent', 'transformer',
                       'attention', 'generative', 'adversarial', 'autoencoder', 'embedding',
                       'tokenization', 'sentiment', 'analysis', 'prediction', 'forecasting',
                       'healthcare', 'medical', 'clinical', 'diagnosis', 'treatment', 'therapy',
                       'biomedical', 'genomics', 'protein', 'drug', 'pharmaceutical', 'clinical',
                       'patient', 'disease', 'cancer', 'cardiovascular', 'neural', 'neurological',
                       'psychology', 'cognitive', 'behavioral', 'social', 'economic', 'finance',
                       'market', 'trading', 'portfolio', 'risk', 'optimization', 'derivative',
                       'cryptocurrency', 'blockchain', 'distributed', 'ledger', 'consensus',
                       'smart', 'contract', 'decentralized', 'token', 'mining', 'proof',
                       'climate', 'change', 'environment', 'sustainable', 'renewable', 'energy',
                       'carbon', 'emission', 'greenhouse', 'global', 'warming', 'pollution',
                       'conservation', 'ecosystem', 'biodiversity', 'species', 'habitat',
                       'quantum', 'physics', 'particle', 'relativity', 'cosmology', 'black',
                       'hole', 'string', 'theory', 'nuclear', 'fusion', 'fission', 'plasma'}

        # Boost domain terms to front of list
        domain_keywords = [word for word in keywords if word in domain_terms]
        other_keywords = [word for word in keywords if word not in domain_terms]
        keywords = domain_keywords + other_keywords

        # Limit to most relevant keywords (top 5-8)
        keywords = keywords[:8]

        # Debug: uncomment to see extracted keywords
        # print(f"DEBUG: Extracted keywords: {keywords}")

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

        logger.info(f"Fallback extraction result: phrases={phrases}, keywords={keywords[:5]}")
        return parsed

    def _clean_extracted_terms(self, parsed: dict) -> dict:
        """
        Post-process Gemini extraction to filter noise and prioritize domain terms
        """
        # Stop words to remove from keywords
        stop_words = {'paper', 'papers', 'research', 'study', 'studies', 'article', 'articles',
                     'find', 'search', 'looking', 'need', 'want', 'about', 'related',
                     'writing', 'work', 'working', 'thesis', 'dissertation', 'source', 'sources'}

        # Filter keywords
        keywords = parsed.get('keywords', [])
        keywords = [k for k in keywords if k.lower() not in stop_words and (len(k) > 2 or k.lower() in VALID_SHORT_TERMS)]

        # Limit keywords to top 5
        keywords = keywords[:5]

        parsed['keywords'] = keywords

        # Phrases are usually good, but limit to top 3
        phrases = parsed.get('phrases', [])
        phrases = phrases[:3]
        parsed['phrases'] = phrases

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
            if term.lower() not in ['ai', 'ml', 'cs', 'it', 'vr', 'ar', 'io', 'ui', 'ux', 'os', 'iq']:
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

        # Search all APIs
        arxiv_results = arxiv_client.search(parsed_query)
        pubmed_results = pubmed_client.search(parsed_query)
        semantic_scholar_results = semantic_scholar_client.search(parsed_query)

        all_results = arxiv_results + pubmed_results + semantic_scholar_results

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