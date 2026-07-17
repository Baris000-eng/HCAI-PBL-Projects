from django.contrib import admin
from .models import PenguinObservation

@admin.register(PenguinObservation)
class PenguinObservationAdmin(admin.ModelAdmin):
    # Display the fields in the admin list view
    list_display = ('id', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'predicted_species', 'created_at')
    # Make the fields filterable
    list_filter = ('predicted_species',)
    # Make the fields editable in the list view
    list_editable = ('predicted_species',)