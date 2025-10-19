# search/services/api_clients.py
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time


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

            print(f"🔍 ArXiv query: {search_query}")  # Debug

            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_arxiv_response(response.text, year)

            return papers

        except requests.RequestException as e:
            print(f"ArXiv API error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in ArXiv client: {e}")
            return []

    def _build_search_query(self, parsed_query: dict) -> str:
        """
        Build intelligent search query using intent, entities, and keywords
        """
        # Get all available data
        keywords = parsed_query.get('keywords', [])
        entities = parsed_query.get('entities', [])
        intent = parsed_query.get('intent', {})
        primary_intent = intent.get('primary_intent', 'find specific papers')

        search_terms = []

        # 1. Add entities (these are often the most important concepts)
        for entity in entities:
            entity_text = entity.get('word', '')
            if entity_text and len(entity_text) > 2:
                search_terms.append(entity_text)

        # 2. Add keywords that aren't already in entities
        entity_words = [e.get('word', '').lower() for e in entities]
        for keyword in keywords:
            if keyword.lower() not in entity_words:
                search_terms.append(keyword)

        # 3. If we have nothing, fall back to original query
        if not search_terms:
            original = parsed_query.get('original_query', '')
            if original:
                search_terms = original.split()

        # Build query based on intent
        if primary_intent == 'literature review':
            # For reviews, search in abstract and title
            query = f"abs:({' AND '.join(search_terms)})"
        elif primary_intent == 'recent developments':
            # Prioritize recent papers
            query = f"all:({' AND '.join(search_terms)})"
        else:
            # Default: search all fields
            query = f"all:({' AND '.join(search_terms)})"

        return query

    def _adjust_max_results(self, parsed_query: dict) -> int:
        """Adjust number of results based on intent"""
        intent = parsed_query.get('intent', {})
        primary_intent = intent.get('primary_intent', '')

        # More results for reviews
        if 'review' in primary_intent or 'survey' in primary_intent or 'comprehensive' in primary_intent:
            return 20
        # Fewer results for specific searches
        elif 'specific' in primary_intent:
            return 5
        # Default
        return self.max_results

    def _determine_sort_order(self, parsed_query: dict) -> str:
        """Determine sort order based on intent"""
        intent = parsed_query.get('intent', {})
        primary_intent = intent.get('primary_intent', '')

        # Recent papers sort by date
        if 'recent' in primary_intent or 'developments' in primary_intent:
            return 'submittedDate'
        # Highly cited would need custom handling (arXiv doesn't support citation sorting)
        elif 'cited' in primary_intent:
            return 'relevance'  # Best we can do
        # Default to relevance
        return 'relevance'

    def _parse_arxiv_response(self, xml_content: str, year_filter: int = None) -> List[Dict]:
        """Parse arXiv XML response into structured paper data"""
        papers = []

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

                # Apply year filter if specified
                if year_filter and pub_year and pub_year != year_filter:
                    continue

                # Build paper object
                paper = {
                    'title': title.text.strip() if title is not None else 'No title',
                    'authors': ', '.join(author_names) if author_names else 'Unknown',
                    'abstract': summary.text.strip() if summary is not None else 'No abstract available',
                    'source': 'arxiv',
                    'year': pub_year,
                    'link': link.text if link is not None else '#'
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

            print(f"🔍 PubMed query: {search_term}")  # Debug

            # Step 1: Search for PMIDs
            pmids = self._search_pubmed(search_term)

            if not pmids:
                return []

            # Step 2: Fetch paper details
            papers = self._fetch_paper_details(pmids)

            return papers

        except Exception as e:
            print(f"PubMed API error: {e}")
            return []

    def _build_search_query(self, parsed_query: dict) -> str:
        """
        Build intelligent PubMed search query using intent, entities, and keywords
        """
        keywords = parsed_query.get('keywords', [])
        entities = parsed_query.get('entities', [])
        intent = parsed_query.get('intent', {})
        filters = parsed_query.get('filters', {})
        year = filters.get('year')
        primary_intent = intent.get('primary_intent', 'find specific papers')

        search_terms = []

        # Add entities first (usually most important)
        for entity in entities:
            entity_text = entity.get('word', '')
            entity_type = entity.get('entity_group', '')

            if entity_text and len(entity_text) > 2:
                # For organizations, add to search
                if entity_type == 'ORG':
                    search_terms.append(f'({entity_text}[Affiliation])')
                # For people (potential authors)
                elif entity_type == 'PER':
                    search_terms.append(f'({entity_text}[Author])')
                # For topics/methods
                else:
                    search_terms.append(entity_text)

        # Add keywords not already covered by entities
        entity_words = [e.get('word', '').lower() for e in entities]
        for keyword in keywords:
            if keyword.lower() not in entity_words:
                search_terms.append(keyword)

        # Fallback to original query
        if not search_terms:
            original = parsed_query.get('original_query', '')
            if original:
                search_terms = original.split()

        # Combine search terms
        query = ' AND '.join(search_terms)

        # Add intent-specific filters
        if 'review' in primary_intent:
            query += ' AND (Review[Publication Type] OR systematic[Title/Abstract])'
        elif 'empirical' in primary_intent:
            query += ' AND (Clinical Trial[Publication Type] OR randomized[Title/Abstract])'

        # Add year filter
        if year:
            query += f' AND {year}[pdat]'

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
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)

                author_str = ', '.join(authors) if authors else 'Unknown'
                if len(author_list) > 5:
                    author_str += ' et al.'

                # Extract publication year
                pub_date = article.find('.//PubDate/Year')
                year = int(pub_date.text) if pub_date is not None else None

                # Extract PMID for link
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else None
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