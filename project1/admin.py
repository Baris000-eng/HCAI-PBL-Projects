from django.contrib import admin
from .models import UploadedFile, TrainingResult

@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at')
    list_filter = ('uploaded_at',)


@admin.register(TrainingResult)
class TrainingResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'model_type', 'split_ratio', 'accuracy', 'f1_score', 'created_at')
    list_filter = ('model_type', 'created_at')
    search_fields = ('model_type',)