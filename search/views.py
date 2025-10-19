from django.shortcuts import render
from django.http import JsonResponse
from .services.query_parser import query_processor
from .services.api_clients import arxiv_client, pubmed_client


def home(request):
    return render(request, 'search/home.html')


def search_api(request):
    # Handle both GET and POST for flexibility
    try:
        if request.method == 'GET':
            query = request.GET.get('q', '').strip()
        elif request.method == 'POST':
            query = request.POST.get('q', '').strip()
        else:
            return JsonResponse({'error': 'Method not allowed'}, status=405)

        if not query:
            return JsonResponse({'error': 'Query parameter is required'}, status=400)

            # Use our NLP processor to understand the query
        parsed_query = query_processor.parse_query(query)

        print(f"Original query: {query}")  # Debug
        print(f"Parsed query: {parsed_query}")  # Debug

        # Search both APIs
        arxiv_results = arxiv_client.search(parsed_query)
        pubmed_results = pubmed_client.search(parsed_query)

        all_results = arxiv_results + pubmed_results

        # Return results
        if not all_results:
            return render(request, 'search/no_results.html', {'query': query})

        return render(request, 'search/results_partial.html', {
            'results': all_results,
            'query': query
        })

    except Exception as e:
        # Log the error in production
        print(f"Error processing search: {e}")
        return JsonResponse({'error': 'An error occurred processing your request'}, status=500)