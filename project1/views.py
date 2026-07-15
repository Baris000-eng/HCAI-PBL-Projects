from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json 
import io 

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from .get_model_type import get_machine_learning_model

def index(request):
    if request.method == "POST" and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)
        
        request.session['df_json'] = df.to_json()
        
        return redirect('project1:visualize')

    return render(request, "project1/index.html")


def visualize(request):

    df_json = request.session.get('df_json')
    
    if not df_json:
        return redirect('project1:upload') 

    df = pd.read_json(io.StringIO(df_json))
    
    chart_data = []
    for i in range(min(len(df), 200)): 
        chart_data.append({'x': float(df.iloc[i, 0]), 'y': float(df.iloc[i, 1])})

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    context = {
        'chart_data': json.dumps(chart_data),
        'columns': numeric_columns,
        'full_data': json.dumps(df.values.tolist()) 
    }
    
    return render(request, "project1/visualize.html", context)

def train(request):
    context = {}
    df_json = request.session.get('df_json')
    if request.method == "POST":
        if df_json:
            df = pd.read_json(io.StringIO(df_json))
            split_ratio = request.POST.get('split_ratio', 0.2)
            test_set_ratio = float(split_ratio)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            selected_metrics = request.POST.getlist('eval_metrics')

            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            model_type = request.POST.get('model_type')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio, random_state=42)

            seed_val = request.POST.get('random_state')
            is_random = request.POST.get('none_checkbox')

            if is_random or not seed_val:
                final_seed = None
            else:
                final_seed = int(seed_val)

            weight_selection = request.POST.get('class_weight')
            final_weight = None if weight_selection == 'none' else 'balanced'

            number_of_epochs = int(request.POST.get('iterations', 100))
            model = get_machine_learning_model(
                model_type=model_type,
                final_seed=final_seed,
                final_weight=final_weight,
                number_of_epochs=number_of_epochs
            )
            
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)

            if 'accuracy' in selected_metrics:
                context['accuracy'] = accuracy_score(y_test, y_pred)

            if 'precision' in selected_metrics: 
                context['precision'] = precision_score(y_test, y_pred, average='weighted')

            if 'recall' in selected_metrics: 
                context['recall'] = recall_score(y_test, y_pred, average='weighted')
            if 'f1_score' in selected_metrics:
                context['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            if 'classification_report' in selected_metrics:
                context['classification_report'] = classification_report(y_test, y_pred)

            if 'confusion_matrix' in selected_metrics: 
                context['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

            context['model_run'] = True

            return render(request, "project1/results.html", context)
            
    return render(request, "project1/train.html")

# Create your views here.
