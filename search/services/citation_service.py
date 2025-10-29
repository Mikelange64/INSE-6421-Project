# search/services/citation_service.py
from typing import Dict
from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


class CitationFormatter:
    """
    Formats paper citations in various academic styles
    """

    def format_citation(self, paper: Dict, style: str = 'apa') -> str:
        """
        Format a paper citation in the specified style

        Args:
            paper: Paper dictionary with title, authors, year, etc.
            style: Citation style ('apa', 'mla', 'chicago', 'ieee')

        Returns:
            Formatted citation string
        """
        style = style.lower()
        logger.debug(f"Formatting citation in {style.upper()} style for paper: {paper.get('title', 'Unknown')[:50]}...")

        try:
            if style == 'apa':
                return self._format_apa(paper)
            elif style == 'mla':
                return self._format_mla(paper)
            elif style == 'chicago':
                return self._format_chicago(paper)
            elif style == 'ieee':
                return self._format_ieee(paper)
            else:
                return self._format_apa(paper)  # Default to APA
        except Exception as e:
            logger.error(f"Citation formatting error for style {style}: {e}")
            return f"Error formatting citation: {paper.get('title', 'Unknown')}"

    def _format_apa(self, paper: Dict) -> str:
        """
        APA 7th Edition format
        Author, A. A. (Year). Title of article. Source. URL
        """
        authors = paper.get('authors', 'Unknown')
        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'Untitled')
        source = paper.get('source', 'Unknown').upper()
        link = paper.get('link', '')

        authors_formatted = self._format_authors_apa(authors)

        citation = f"{authors_formatted} ({year}). {title}. {source}."

        if link:
            citation += f" {link}"

        return citation

    def _format_mla(self, paper: Dict) -> str:
        """
        MLA 9th Edition format
        Author(s). "Title of Article." Source, Year, URL.
        """
        authors = paper.get('authors', 'Unknown')
        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'Untitled')
        source = paper.get('source', 'Unknown').upper()
        link = paper.get('link', '')

        authors_formatted = self._format_authors_mla(authors)

        citation = f'{authors_formatted}. "{title}." {source}, {year}.'

        if link:
            citation += f" {link}."

        return citation

    def _format_chicago(self, paper: Dict) -> str:
        """
        Chicago style (Notes and Bibliography)
        Author(s). "Title of Article." Source (Year). URL.
        """
        authors = paper.get('authors', 'Unknown')
        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'Untitled')
        source = paper.get('source', 'Unknown').upper()
        link = paper.get('link', '')

        citation = f'{authors}. "{title}." {source} ({year}).'

        if link:
            citation += f" {link}."

        return citation

    def _format_ieee(self, paper: Dict) -> str:
        """
        IEEE style
        [#] Author(s), "Title of article," Source, Year. [Online]. Available: URL
        """
        authors = paper.get('authors', 'Unknown')
        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'Untitled')
        source = paper.get('source', 'Unknown').upper()
        link = paper.get('link', '')

        authors_formatted = self._format_authors_ieee(authors)

        citation = f'{authors_formatted}, "{title}," {source}, {year}.'

        if link:
            citation += f' [Online]. Available: {link}'

        return citation

    def _format_authors_apa(self, authors: str) -> str:
        """Format authors for APA style"""
        if not authors or authors == 'Unknown':
            return 'Unknown'

        if 'et al.' in authors.lower():
            return authors

        author_list = self._split_authors(authors)

        if len(author_list) == 0:
            return 'Unknown'
        elif len(author_list) == 1:
            return author_list[0]
        elif len(author_list) == 2:
            return f"{author_list[0]}, & {author_list[1]}"
        else:
            return f"{author_list[0]} et al."

    def _format_authors_mla(self, authors: str) -> str:
        """Format authors for MLA style"""
        if not authors or authors == 'Unknown':
            return 'Unknown'

        if 'et al.' in authors.lower():
            return authors

        author_list = self._split_authors(authors)

        if len(author_list) == 0:
            return 'Unknown'
        elif len(author_list) == 1:
            return author_list[0]
        elif len(author_list) == 2:
            return f"{author_list[0]}, and {author_list[1]}"
        else:
            return f"{author_list[0]}, et al."

    def _format_authors_ieee(self, authors: str) -> str:
        """Format authors for IEEE style"""
        if not authors or authors == 'Unknown':
            return 'Unknown'

        return authors

    def _split_authors(self, authors: str) -> list:
        """Split author string into list"""
        # If authors already contains et al., extract authors before et al.
        if 'et al.' in authors.lower():
            # Find where et al. starts and extract everything before it
            et_al_pos = authors.lower().find('et al.')
            authors = authors[:et_al_pos].strip()
            # Remove trailing comma if present
            authors = authors.rstrip(',').strip()
        
        authors = authors.replace(' and ', ', ')
        authors = authors.replace(' & ', ', ')

        author_list = [a.strip() for a in authors.split(',') if a.strip()]
        # Remove any leftover 'et al.' mentions (shouldn't happen, but just in case)
        author_list = [a for a in author_list if 'et al.' not in a.lower()]

        return author_list


# Singleton instance
citation_formatter = CitationFormatter()
