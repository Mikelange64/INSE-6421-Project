# search/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .services.conversational_agent import research_agent
import uuid


def home(request):
    """Home page with chat interface"""
    # Generate session ID for this user
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())

    return render(request, 'search/home.html', {
        'session_id': request.session['session_id']
    })


@require_http_methods(["POST"])
def chat_api(request):
    """
    Main chat endpoint - handles conversational messages
    """
    try:
        # Get message and session
        user_message = request.POST.get('message', '').strip()
        session_id = request.session.get('session_id', 'default')

        if not user_message:
            return JsonResponse({
                'error': 'Message is required'
            }, status=400)

        # Process message through agent
        response = research_agent.chat(user_message, session_id)

        # Return response
        return JsonResponse({
            'agent_response': response['agent_response'],
            'action': response['action'],
            'data': response.get('data', {})
        })

    except Exception as e:
        print(f"Chat API error: {e}")
        return JsonResponse({
            'error': 'An error occurred processing your message',
            'details': str(e)
        }, status=500)


@require_http_methods(["POST"])
def get_citations_api(request):
    """
    Get citations for specific papers
    """
    try:
        session_id = request.session.get('session_id', 'default')
        paper_indices = request.POST.getlist('indices[]')
        style = request.POST.get('style', 'apa')

        # Convert indices to integers
        paper_indices = [int(i) for i in paper_indices]

        # Get citations
        citations = research_agent.get_citations(session_id, paper_indices, style)

        return JsonResponse({
            'citations': citations
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["POST"])
def clear_conversation_api(request):
    """
    Clear conversation history
    """
    try:
        session_id = request.session.get('session_id', 'default')
        research_agent.clear_session(session_id)

        return JsonResponse({
            'success': True,
            'message': 'Conversation cleared'
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)