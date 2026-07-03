from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/metrics/', views.get_metrics, name='get_metrics'),
    path('api/learning_to_defer/', views.learning_to_defer, name='learning_to_defer'),
    path('api/active_learning/next/', views.al_next_sample, name='al_next_sample'),
    path('api/active_learning/query/', views.al_query, name='al_query'),
    path('api/plot-metrics/', views.trigger_plot_metrics, name='api_plot_metrics'),
    path('api/clear-plot/', views.clear_plot, name='api_clear_plot')
]