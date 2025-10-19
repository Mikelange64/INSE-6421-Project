# search/services/api_clients.py

class ArXivClient:
    def search(self, parsed_query: dict) -> list:
        """
        Mock ArXiv search - returns fake papers for testing
        """
        print(f"🔍 Mock searching ArXiv for: {parsed_query['keywords']}")

        # Return realistic mock data
        return [
            {
                'title': f"Research Paper on {' '.join(parsed_query['keywords'])}",
                'authors': 'Researcher A, Researcher B',
                'abstract': f"This paper explores {' '.join(parsed_query['keywords'])} using advanced methodologies...",
                'source': 'arxiv',
                'link': 'https://arxiv.org/abs/0000.00001',
                'year': parsed_query['filters'].get('year', 2024)
            }
        ]

class PubMedClient:
    def search(self, parsed_query: dict) -> list:
        """
        Mock PubMed search - returns fake papers for testing
        """
        print(f"🔍 Mock searching PubMed for: {parsed_query['keywords']}")

        return [
            {
                'title': f"Medical Study: {' '.join(parsed_query['keywords'])} in Healthcare",
                'authors': 'Dr. Smith, Dr. Johnson',
                'abstract': f"This study investigates the applications of {' '.join(parsed_query['keywords'])} in clinical settings...",
                'source': 'pubmed',
                'link': 'https://pubmed.ncbi.nlm.nih.gov/00000001/',
                'year': parsed_query['filters'].get('year', 2024)
            }
        ]


# Create singleton instances
arxiv_client = ArXivClient()
pubmed_client = PubMedClient()