from django.conf import settings
from django.shortcuts import render
from django.http import FileResponse, JsonResponse
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import os
import json


file_path = os.path.join(settings.BASE_DIR, 'project4', 'movie_metadata.csv')

def load_and_process_local_dataset(file_path=file_path):
    dataset = load_dataset('csv', data_files=file_path)
    df = dataset['train'].to_pandas()
    df = df.dropna(subset=['movie_title', 'genres', 'imdb_score'])
    
    cv = CountVectorizer(tokenizer=lambda text: text.split('|'), lowercase=False)
    genre_matrix = cv.fit_transform(df['genres']).toarray()
    
    scaler = MinMaxScaler()
    scaled_scores = scaler.fit_transform(df[['imdb_score']])
    
    feature_vectors = np.hstack((genre_matrix, scaled_scores))
    
    movies = []
    for i, row in df.reset_index(drop=True).iterrows():
        movies.append({
            'id': i + 1,
            'title': str(row['movie_title']).strip(),
            'year': int(row['title_year']) if pd.notna(row['title_year']) else 2000,
            'genre': str(row['genres']),
            'score': float(row['imdb_score']),
            'feature_vector': feature_vectors[i].tolist()
        })
    return movies

ALL_MOVIES = load_and_process_local_dataset(file_path)
D = len(ALL_MOVIES[0]['feature_vector']) if ALL_MOVIES else 10

def get_random_movies_from_dataset(n=10):
    return random.sample(ALL_MOVIES, min(n, len(ALL_MOVIES)))

def get_or_init_user_vector(request):
    if 'user_w' not in request.session:
        request.session['user_w'] = np.zeros(D).tolist()
    return np.array(request.session['user_w'])

def index(request):
    return render(request, 'project4/index.html')

def design1_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        chosen_id = int(data.get('chosen_id'))
        other_id = int(data.get('other_id'))
        
        w = get_or_init_user_vector(request)
        
        x_a = np.array(next(m['feature_vector'] for m in ALL_MOVIES if m['id'] == chosen_id))
        x_b = np.array(next(m['feature_vector'] for m in ALL_MOVIES if m['id'] == other_id))

        # Update user preference vector using Bradley-Terry model 
        lr_val = 0.05 
        w = bradley_terry_update(w, x_a, x_b, lr=lr_val)
        
        request.session['user_w'] = w.tolist()
        return JsonResponse({'status': 'success', 'updated_w': w.tolist()})
        
    context = {'movies': get_random_movies_from_dataset(2)}
    return render(request, 'project4/design1.html', context)

def design2_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        ranked_ids = [int(i) for i in data.get('ranked_ids')]
        
        w = get_or_init_user_vector(request)
        lr_val = 0.005
        
        # Compute utility scores dictionary for all movies
        utility_scores = {m['id']: np.dot(w, m['feature_vector']) for m in ALL_MOVIES}
        
        # Call plackett_luce_probability using ranked_ids and utility_scores
        ranking_prob = plackett_luce_probability(ranked_ids, utility_scores)
        
        current_pool = [np.array(next(m['feature_vector'] for m in ALL_MOVIES if m['id'] == mid)) for mid in ranked_ids]
        
        grad = np.zeros_like(w)
        for k in range(len(current_pool)):
            x_k = current_pool[k]
            remaining_pool = current_pool[k:]
            
            exp_utilities = np.array([np.exp(np.dot(w, x)) for x in remaining_pool])
            sum_exp = np.sum(exp_utilities)
            
            expected_x = np.sum([exp_utilities[j] * remaining_pool[j] for j in range(len(remaining_pool))], axis=0) / sum_exp
            grad += (x_k - expected_x)
            
        w += lr_val * grad
        request.session['user_w'] = w.tolist()
        return JsonResponse({
            'status': 'success', 
            'ranking_probability': float(ranking_prob),
            'updated_w': w.tolist()
        })
        
    context = {'movies': get_random_movies_from_dataset(10)}
    return render(request, 'project4/design2.html', context)

def download_pdf(request):
    pdf_path = os.path.join('static', 'pdf', 'project_report.pdf')
    return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

def bradley_terry_update(w, x_a, x_b, lr=0.05):
    u_a = np.dot(w, x_a)
    u_b = np.dot(w, x_b)
    
    exp_a = np.exp(u_a)
    exp_b = np.exp(u_b)
    
    grad = x_a - (exp_a * x_a + exp_b * x_b) / (exp_a + exp_b)
    return w + lr * grad

def plackett_luce_probability(ranked_ids, utility_scores_dict):
    prob = 1.0
    current_pool = list(ranked_ids)
    for i in ranked_ids:
        exp_utilities = np.array([np.exp(utility_scores_dict[j]) for j in current_pool])
        prob *= np.exp(utility_scores_dict[i]) / np.sum(exp_utilities)
        current_pool.remove(i)
    return prob