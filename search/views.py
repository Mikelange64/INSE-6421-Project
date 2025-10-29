# search/views.py
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .services.conversational_agent import research_agent
from .services.logging_config import get_logger
from .models import ChatHistory, PaperHistory, SharedConversation
import uuid
import time
import json

# Initialize logger
logger = get_logger(__name__)


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
    start_time = time.time()
    
    try:
        # Get message and session
        user_message = request.POST.get('message', '').strip()
        session_id = request.session.get('session_id', 'default')

        logger.info(f"Chat API request - Session: {session_id[:8]}... - Message length: {len(user_message)} chars")

        if not user_message:
            logger.warning("Empty message received")
            return JsonResponse({
                'error': 'Message is required'
            }, status=400)

        # Process message through agent
        response = research_agent.chat(user_message, session_id)

        response_time = time.time() - start_time
        logger.debug(f"Request processed in {response_time:.2f}s - Action: {response.get('action', 'unknown')}")

        # Save to chat history
        try:
            ChatHistory.objects.create(
                session_id=session_id,
                user_message=user_message,
                agent_response=response['agent_response'],
                action_type=response['action']
            )
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")

        # Return response
        return JsonResponse({
            'agent_response': response['agent_response'],
            'action': response['action'],
            'data': response.get('data', {})
        })

    except Exception as e:
        logger.error(f"Chat API error: {e}", exc_info=True)
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


@require_http_methods(["GET"])
def get_chat_history_api(request):
    """
    Retrieve chat history for current session
    """
    try:
        session_id = request.session.get('session_id', 'default')
        limit = int(request.GET.get('limit', 50))
        
        history = ChatHistory.objects.filter(session_id=session_id)[:limit]
        
        history_data = [{
            'user_message': chat.user_message,
            'agent_response': chat.agent_response,
            'action_type': chat.action_type,
            'timestamp': chat.timestamp.isoformat()
        } for chat in history]
        
        return JsonResponse({
            'history': history_data,
            'count': len(history_data)
        })
    
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}", exc_info=True)
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_paper_history_api(request):
    """
    Retrieve paper history for current session
    """
    try:
        session_id = request.session.get('session_id', 'default')
        limit = int(request.GET.get('limit', 50))
        
        history = PaperHistory.objects.filter(session_id=session_id)[:limit]
        
        history_data = [{
            'title': paper.paper_title,
            'link': paper.paper_link,
            'data': paper.paper_data,
            'timestamp': paper.timestamp.isoformat()
        } for paper in history]
        
        return JsonResponse({
            'history': history_data,
            'count': len(history_data)
        })
    
    except Exception as e:
        logger.error(f"Error fetching paper history: {e}", exc_info=True)
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["POST"])
def save_paper_click_api(request):
    """
    Save when user clicks 'Read Paper'
    """
    try:
        session_id = request.session.get('session_id', 'default')
        paper_data = json.loads(request.body)
        
        PaperHistory.objects.create(
            session_id=session_id,
            paper_title=paper_data.get('title', 'Unknown'),
            paper_link=paper_data.get('link', ''),
            paper_data=paper_data
        )
        
        return JsonResponse({
            'success': True,
            'message': 'Paper saved to history'
        })
    
    except Exception as e:
        logger.error(f"Error saving paper click: {e}", exc_info=True)
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["POST"])
def share_conversation_api(request):
    """
    Create a shareable link for current conversation
    """
    try:
        session_id = request.session.get('session_id', 'default')
        
        # Get chat history
        history = ChatHistory.objects.filter(session_id=session_id)[:50]
        conversation_data = [{
            'user_message': chat.user_message,
            'agent_response': chat.agent_response,
            'timestamp': chat.timestamp.isoformat()
        } for chat in history]
        
        # Create share ID
        share_id = str(uuid.uuid4())
        
        # Save to database
        SharedConversation.objects.create(
            share_id=share_id,
            session_id=session_id,
            conversation_data=conversation_data
        )
        
        # Generate shareable URL
        share_url = request.build_absolute_uri(f'/shared/{share_id}')
        
        return JsonResponse({
            'success': True,
            'share_url': share_url,
            'share_id': share_id
        })
    
    except Exception as e:
        logger.error(f"Error creating share link: {e}", exc_info=True)
        return JsonResponse({
            'error': str(e)
        }, status=500)


def view_shared_conversation(request, share_id):
    """
    View a shared conversation (read-only)
    """
    try:
        shared = get_object_or_404(SharedConversation, share_id=share_id)
        
        # Increment view count
        shared.view_count += 1
        shared.save()
        
        return render(request, 'search/shared_conversation.html', {
            'conversation': shared.conversation_data,
            'created_at': shared.created_at,
            'view_count': shared.view_count
        })
    
    except Exception as e:
        logger.error(f"Error viewing shared conversation: {e}", exc_info=True)
        return render(request, 'search/shared_conversation.html', {
            'error': 'Conversation not found'
        })