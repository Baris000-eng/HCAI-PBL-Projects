from django.db import models

class PenguinObservation(models.Model):
    # Features used in the model
    bill_length_mm = models.FloatField()
    bill_depth_mm = models.FloatField()
    flipper_length_mm = models.FloatField()
    body_mass_g = models.FloatField()
    
    # To store the model prediction 
    predicted_species = models.CharField(max_length=100, null=True, blank=True)
    
    # Track when the record was added
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Penguin {self.id} - {self.predicted_species or 'Unclassified'}"