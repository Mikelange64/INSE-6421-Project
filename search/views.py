from django.shortcuts import render
from django.http import JsonResponse
from .services import query_processor, arxiv_client, pubmed_client


def home(request):
    return render(request, 'search/home.html')


def search_api(request):
    # Handle both GET and POST for flexibility
    if request.method == 'GET':
        query = request.GET.get('q', '')
    elif request.method == 'POST':
        query = request.POST.get('q', '')
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

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