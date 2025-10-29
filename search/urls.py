from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/citations/', views.get_citations_api, name='get_citations'),
    path('api/clear/', views.clear_conversation_api, name='clear_conversation'),
]