from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
from django.http import FileResponse, JsonResponse
import random
import os

# Dummy movie generator simulating IMDB 5000 dataset
def get_random_movies(n=10):
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller']
    return [
        {
            'id': i,
            'title': f'Movie Sample {random.randint(100, 999)}',
            'year': random.randint(1990, 2023),
            'genre': random.choice(genres),
            'score': round(random.uniform(5.0, 9.5), 1)
        }
        for i in range(1, n + 1)
    ]

def landing_page(request):
    return render(request, 'landing.html')

def design1_view(request):
    if request.method == 'POST':
        # Handle preference saving 
        return JsonResponse({'status': 'success'})
    context = {'movies': get_random_movies(2)}
    return render(request, 'design1.html', context)

def design2_view(request):
    if request.method == 'POST':
        # Handle ranking vector saving 
        return JsonResponse({'status': 'success'})
    context = {'movies': get_random_movies(10)}
    return render(request, 'design2.html', context)

def download_pdf(request):
    # Path to your uploaded project report PDF
    pdf_path = os.path.join('static', 'pdf', 'project_report.pdf')
    return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')