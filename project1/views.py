from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import json 

def index(request):
    context = {}
    if request.method == "POST" and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)
        
        chart_data = []
        for i in range(min(len(df), 150)): 
            chart_data.append({'x': float(df.iloc[i, 0]), 'y': float(df.iloc[i, 1])})
        
        context['chart_data'] = json.dumps(chart_data)
        context['columns'] = df.columns.tolist()
        
        request.session['df_json'] = df.to_json()

    return render(request, "project1/index.html", context)

def train(request):
    context = {}
    if request.method == "POST":
        df_json = request.session.get('df_json')
        if df_json:
            df = pd.read_json(df_json)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            model_type = request.POST.get('model_type')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            if model_type.lower() == 'rfc':
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)
            
            context['score'] = model.score(X_test, y_test)
            context['model_run'] = True
            
    return render(request, "project1/index.html", context)

# Create your views here.
