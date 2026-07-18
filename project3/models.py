from django.db import models

class NewsArticle(models.Model):
    # Categories: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
    CATEGORY_CHOICES = [
        (0, 'World'),
        (1, 'Sports'),
        (2, 'Business'),
        (3, 'Sci/Tech'),
    ]

    text = models.TextField()
    label = models.IntegerField(choices=CATEGORY_CHOICES, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_label_display()} - {self.text[:50]}..."