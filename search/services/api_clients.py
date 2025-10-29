# search/services/api_clients.py
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time
from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


class ArxivClient:
    """
    Client for arXiv API
    Documentation: https://info.arxiv.org/help/api/index.html
    """

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_results = 10

    def search(self, parsed_query: dict) -> List[Dict]:
        """
        Search arXiv using parsed query data with AI-enhanced understanding

        Args:
            parsed_query: Dict containing:
                - 'keywords': List of extracted keywords
                - 'filters': Dict with year, etc.
                - 'intent': Dict with primary_intent, confidence
                - 'entities': List of NER entities
                - 'original_query': Original user query

        Returns:
            List of paper dictionaries
        """
        try:
            # Build intelligent search query using ALL available data
            search_query = self._build_search_query(parsed_query)

            if not search_query:
                return []

            # Extract filters
            filters = parsed_query.get('filters', {})
            year = filters.get('year')

            # Prepare API parameters
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': self._adjust_max_results(parsed_query),
                'sortBy': self._determine_sort_order(parsed_query),
                'sortOrder': 'descending'
            }

            logger.debug(f"ArXiv query: {search_query}")
            logger.debug(f"ArXiv params: max_results={params['max_results']}, sortBy={params['sortBy']}")

            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_arxiv_response(response.text, filters)

            logger.info(f"ArXiv returned {len(papers)} papers")
            return papers

        except requests.RequestException as e:
            logger.error(f"ArXiv API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in ArXiv client: {e}", exc_info=True)
            return []

    def _build_search_query(self, parsed_query: dict) -> str:
        """
        Build search query using phrases and keywords
        """
        all_terms = parsed_query.get('all_terms', [])

        if not all_terms:
            # Fallback to keywords
            all_terms = parsed_query.get('keywords', [])

        if not all_terms:
            return ''

        # Build query - phrases are already in all_terms
        search_terms = []
        for term in all_terms:
            # If term has spaces, it's a phrase - quote it
            if ' ' in term:
                search_terms.append(f'"{term}"')
            else:
                search_terms.append(term)

        return 'all:(' + ' AND '.join(search_terms) + ')'

    def _adjust_max_results(self, parsed_query: dict) -> int:
        """Adjust number of results based on intent"""
        # intent is a string, not a dict
        intent_str = str(parsed_query.get('intent', ''))

        # More results for reviews
        if 'review' in intent_str or 'survey' in intent_str or 'comprehensive' in intent_str:
            return 20
        # Fewer results for specific searches
        elif 'specific' in intent_str:
            return 5
        # Default
        return self.max_results

    def _determine_sort_order(self, parsed_query: dict) -> str:
        """Determine sort order based on intent"""
        # intent is a string, not a dict
        intent_str = str(parsed_query.get('intent', ''))

        # Recent papers sort by date
        if 'recent' in intent_str or 'developments' in intent_str:
            return 'submittedDate'
        # Highly cited would need custom handling (arXiv doesn't support citation sorting)
        elif 'cited' in intent_str:
            return 'relevance'  # Best we can do
        # Default to relevance
        return 'relevance'

    def _parse_arxiv_response(self, xml_content: str, filters: dict = None) -> List[Dict]:
        """Parse arXiv XML response into structured paper data"""
        papers = []
        filters = filters or {}

        try:
            # Parse XML with namespace handling
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', namespace)

            for entry in entries:
                # Extract paper details
                title = entry.find('atom:title', namespace)
                summary = entry.find('atom:summary', namespace)
                published = entry.find('atom:published', namespace)
                link = entry.find('atom:id', namespace)

                # Extract authors
                authors = entry.findall('atom:author', namespace)
                author_names = []
                for author in authors:
                    name = author.find('atom:name', namespace)
                    if name is not None and name.text:
                        author_names.append(name.text)

                # Extract year from published date
                pub_year = None
                if published is not None and published.text:
                    pub_year = int(published.text[:4])

                # Apply year filters if specified
                year_min = filters.get('year_min')
                year_max = filters.get('year_max')
                single_year = filters.get('year')

                # Check year constraints
                if single_year and pub_year and pub_year != single_year:
                    continue
                elif year_min and pub_year and pub_year < year_min:
                    continue
                elif year_max and pub_year and pub_year > year_max:
                    continue

                # Build paper object
                paper = {
                    'title': title.text.strip() if title is not None and title.text else 'No title',
                    'authors': ', '.join(author_names) if author_names else 'Unknown',
                    'abstract': summary.text.strip() if summary is not None and summary.text else 'No abstract available',
                    'source': 'arxiv',
                    'year': pub_year,
                    'link': link.text.strip() if link is not None and link.text else '#'
                }

                papers.append(paper)

        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
        except Exception as e:
            print(f"Error parsing arXiv response: {e}")

        return papers

class PubMedClient:
    """
    Client for PubMed E-utilities API
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.max_results = 10
        # Rate limiting: NCBI recommends max 3 requests/second without API key
        self.request_delay = 0.34
        self.last_request_time = 0

    def search(self, parsed_query: dict) -> List[Dict]:
        """
        Search PubMed using parsed query data with AI-enhanced understanding

        Args:
            parsed_query: Dict containing intent, entities, keywords, filters

        Returns:
            List of paper dictionaries
        """
        try:
            # Build intelligent search query
            search_term = self._build_search_query(parsed_query)

            if not search_term:
                return []

            logger.debug(f"PubMed query: {search_term}")

            # Step 1: Search for PMIDs
            pmids = self._search_pubmed(search_term)

            if not pmids:
                logger.debug("PubMed returned no PMIDs")
                return []

            logger.debug(f"PubMed found {len(pmids)} PMIDs, fetching details...")

            # Step 2: Fetch paper details
            papers = self._fetch_paper_details(pmids)

            logger.info(f"PubMed returned {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"PubMed API error: {e}", exc_info=True)
            return []

    def _build_search_query(self, parsed_query: dict) -> str:
        """
        Build PubMed search query
        """
        all_terms = parsed_query.get('all_terms', [])
        filters = parsed_query.get('filters', {})

        if not all_terms:
            return ''

        # Build base query
        search_terms = []
        for term in all_terms:
            if ' ' in term:
                search_terms.append(f'"{term}"')
            else:
                search_terms.append(term)

        query = ' AND '.join(search_terms)

        # Add year filters
        year = filters.get('year')
        year_min = filters.get('year_min')
        year_max = filters.get('year_max')

        # Prioritize ranges over single years when both are present
        if year_min and year_max:
            query += f' AND {year_min}:{year_max}[pdat]'
        elif year:
            query += f' AND {year}[pdat]'
        elif year_min:
            query += f' AND {year_min}:3000[pdat]'
        elif year_max:
            query += f' AND 1900:{year_max}[pdat]'

        return query

    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def _search_pubmed(self, search_term: str) -> List[str]:
        """Search PubMed and return PMIDs"""
        self._rate_limit()

        try:
            url = f"{self.base_url}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': search_term,
                'retmax': self.max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])

            return pmids

        except requests.RequestException as e:
            print(f"PubMed search error: {e}")
            return []

    def _fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch full details for a list of PMIDs"""
        if not pmids:
            return []

        self._rate_limit()

        try:
            url = f"{self.base_url}/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            return self._parse_pubmed_response(response.text)

        except requests.RequestException as e:
            print(f"PubMed fetch error: {e}")
            return []

    def _parse_pubmed_response(self, xml_content: str) -> List[Dict]:
        """Parse PubMed XML response into structured paper data"""
        papers = []

        try:
            root = ET.fromstring(xml_content)
            articles = root.findall('.//PubmedArticle')

            for article in articles:
                # Extract title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else 'No title'

                # Extract abstract
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else 'No abstract available'

                # Extract authors
                author_list = article.findall('.//Author')
                authors = []
                for author in author_list[:5]:
                    lastname = author.find('LastName')
                    forename = author.find('ForeName')
                    if lastname is not None and lastname.text:
                        name = lastname.text.strip()
                        if forename is not None and forename.text:
                            name = f"{forename.text.strip()} {name}"
                        authors.append(name)

                author_str = ', '.join(authors) if authors else 'Unknown'
                if len(author_list) > 5:
                    author_str += ' et al.'

                # Extract publication year
                pub_date = article.find('.//PubDate/Year')
                year = None
                if pub_date is not None and pub_date.text:
                    try:
                        year = int(pub_date.text.strip())
                    except ValueError:
                        year = None

                # Extract PMID for link
                pmid_elem = article.find('.//PMID')
                pmid = None
                if pmid_elem is not None and pmid_elem.text:
                    pmid = pmid_elem.text.strip()
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '#'

                paper = {
                    'title': title,
                    'authors': author_str,
                    'abstract': abstract,
                    'source': 'pubmed',
                    'year': year,
                    'link': link
                }

                papers.append(paper)

        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
        except Exception as e:
            print(f"Error parsing PubMed response: {e}")

        return papers


# Singleton instances
arxiv_client = ArxivClient()
pubmed_client = PubMedClient()