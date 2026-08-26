from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('study/design1/', views.design1_view, name='design1'),
    path('study/design2/', views.design2_view, name='design2'),
    path('download-report/', views.download_pdf, name='download_pdf')
]