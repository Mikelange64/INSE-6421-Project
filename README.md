# Mayele — AI Research Paper Aggregator

Mayele is a conversational research assistant that translates natural language queries into structured academic paper searches across multiple databases. Ask it a research question and it finds, ranks, and cites relevant papers for you.

## Features

- **Conversational interface** — ask research questions in plain English; Gemini AI handles intent detection and query parsing
- **Multi-source aggregation** — searches arXiv, PubMed, and Semantic Scholar in a single query
- **Relevance ranking** — papers scored 1–5 stars based on title/abstract match, recency, and source authority
- **Citation generation** — export citations in APA, MLA, Chicago, or IEEE format
- **History tracking** — per-session chat history and paper click history
- **Shareable conversations** — generate a read-only link to share a research session

## Tech Stack

- **Backend:** Django 5.2, SQLite
- **AI:** Google Gemini API (intent detection, query understanding)
- **Data Sources:** arXiv API, PubMed API, Semantic Scholar API
- **Frontend:** HTML5/CSS/JS (no framework)

## Project Structure

```
├── ai_paper_aggregator/   # Django project config (settings, URLs, WSGI/ASGI)
├── search/
│   ├── services/          # Business logic
│   │   ├── conversational_agent.py  # Intent detection + query parsing
│   │   ├── api_clients.py           # arXiv, PubMed, Semantic Scholar clients
│   │   ├── ranking_service.py       # 1-5 star relevance scoring
│   │   ├── citation_service.py      # APA, MLA, Chicago, IEEE formatting
│   │   └── logging_config.py        # Rotating file logger
│   ├── templates/         # HTML templates
│   ├── static/            # Images and assets
│   ├── models.py          # ChatHistory, PaperHistory, SharedConversation
│   ├── views.py           # API endpoints and page views
│   └── urls.py            # App URL routing
├── manage.py
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/Mikelane64/INSE-6421-Project.git
cd mayele

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
DEBUG=True
GEMINI_API_KEY=your_gemini_api_key_here
HF_API_TOKEN=your_huggingface_token_here  # optional
```

### Run

```bash
python manage.py migrate
python manage.py runserver
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## How It Works

1. User submits a natural language query (e.g. *"recent papers on transformer efficiency"*)
2. Gemini classifies the intent — **SEARCH** or general **CONVERSATION**
3. For search queries, key terms and filters (year, topic) are extracted
4. arXiv, PubMed, and Semantic Scholar are queried in parallel
5. Results are scored and ranked by relevance
6. Papers are returned with optional citation export

## License

MIT
