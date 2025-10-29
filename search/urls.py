from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/citations/', views.get_citations_api, name='get_citations'),
    path('api/clear/', views.clear_conversation_api, name='clear_conversation'),
    path('api/chat-history/', views.get_chat_history_api, name='get_chat_history'),
    path('api/paper-history/', views.get_paper_history_api, name='get_paper_history'),
    path('api/save-paper/', views.save_paper_click_api, name='save_paper_click'),
    path('api/share/', views.share_conversation_api, name='share_conversation'),
    path('shared/<str:share_id>/', views.view_shared_conversation, name='view_shared'),
]