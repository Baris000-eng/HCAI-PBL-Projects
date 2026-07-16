from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File {self.id} uploaded at {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"


class TrainingResult(models.Model):
    model_type = models.CharField(max_length=100)
    split_ratio = models.FloatField()
    random_state = models.IntegerField(null=True, blank=True)
    epochs = models.IntegerField(default=100)
    
    # Evaluation Metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    classification_report = models.TextField(null=True, blank=True)
    # Stores matrix as a JSON
    confusion_matrix = models.JSONField(null=True, blank=True)  
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_type} Run - {self.created_at.strftime('%Y-%m-%d %H:%M')}"