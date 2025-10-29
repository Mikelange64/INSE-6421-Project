from django.db import models
from django.utils import timezone


class ChatHistory(models.Model):
    """
    Store conversation history for each user session
    """
    session_id = models.CharField(max_length=255, db_index=True)
    user_message = models.TextField()
    agent_response = models.TextField()
    action_type = models.CharField(max_length=50, default='conversation')  # 'search', 'conversation', 'clarification', etc.
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Chat Histories'
    
    def __str__(self):
        return f"{self.session_id[:8]} - {self.user_message[:50]}"


class PaperHistory(models.Model):
    """
    Store papers that users have clicked 'Read Paper' on
    """
    session_id = models.CharField(max_length=255, db_index=True)
    paper_title = models.TextField()
    paper_link = models.URLField(max_length=500)
    paper_data = models.JSONField()  # Store full paper object (authors, abstract, etc.)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Paper Histories'
    
    def __str__(self):
        return f"{self.session_id[:8]} - {self.paper_title[:50]}"


class SharedConversation(models.Model):
    """
    Store shareable conversation links
    """
    share_id = models.CharField(max_length=36, unique=True, db_index=True)  # UUID
    session_id = models.CharField(max_length=255)
    conversation_data = models.JSONField()  # Store entire conversation
    created_at = models.DateTimeField(default=timezone.now)
    view_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Shared: {self.share_id}"
