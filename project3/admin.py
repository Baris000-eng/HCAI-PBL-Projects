from django.contrib import admin
from .models import NewsArticle

@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    # Columns to display in the main admin table list
    list_display = ('id', 'get_label_display', 'short_text', 'created_at')
    
    # Add a filtering sidebar on the right side for Categories and Dates
    list_filter = ('label', 'created_at')
    
    # Add a search bar at the top to search through the article content
    search_fields = ('text',)
    
    # Order articles by newest first by default
    ordering = ('-created_at',)

    # Helper to keep the text column neat in the admin list view
    def short_text(self, obj):
        return obj.text[:75] + "..." if len(obj.text) > 75 else obj.text
    short_text.short_description = 'Article Content'