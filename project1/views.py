from django.shortcuts import render, redirect, get_object_or_404
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json 

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from .get_model_type import get_machine_learning_model
from .models import UploadedFile, TrainingResult

def index(request):
    if request.method == "POST" and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        # Save the file to the database using the UploadedFile model
        uploaded_file_obj = UploadedFile.objects.create(file=csv_file)
        
        # Save the database ID of the file in the session instead of the entire dataset
        request.session['file_id'] = uploaded_file_obj.id
        
        return redirect('project1:visualize')

    return render(request, "project1/index.html")


def visualize(request):

    file_id = request.session.get('file_id')
    
    if not file_id:
        return redirect('project1:index') 

    # Retrieve the file record from the database
    uploaded_file_obj = get_object_or_404(UploadedFile, id=file_id)
    
    # Read the file from the disk storage using its file path
    df = pd.read_csv(uploaded_file_obj.file.path)
    
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
    file_id = request.session.get('file_id')
    if request.method == "POST":
        if file_id:
            uploaded_file_obj = get_object_or_404(UploadedFile, id=file_id)
            df = pd.read_csv(uploaded_file_obj.file.path)
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

            # Gather metrics values (conditionally check selected_metrics)
            metrics_data = {
                'accuracy': accuracy_score(y_test, y_pred) if 'accuracy' in selected_metrics else None,
                'precision': precision_score(y_test, y_pred, average='weighted') if 'precision' in selected_metrics else None,
                'recall': recall_score(y_test, y_pred, average='weighted') if 'recall' in selected_metrics else None,
                'f1_score': f1_score(y_test, y_pred, average='weighted') if 'f1_score' in selected_metrics else None,
                'classification_report': classification_report(y_test, y_pred) if 'classification_report' in selected_metrics else None,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist() if 'confusion_matrix' in selected_metrics else None,
            }

            # Create database entry for TrainingResult
            result = TrainingResult.objects.create(
                model_type=model_type,
                split_ratio=split_ratio,
                random_state=final_seed,
                epochs=number_of_epochs,
                accuracy=metrics_data['accuracy'],
                precision=metrics_data['precision'],
                recall=metrics_data['recall'],
                f1_score=metrics_data['f1_score'],
                classification_report=metrics_data['classification_report'],
                confusion_matrix=metrics_data['confusion_matrix']
            )

            # Construct context for rendering template
            context = {
                'model_run': True,
                'model_type': result.model_type,
                'split_ratio': result.split_ratio,
                'epochs': result.epochs,
                # Only send active metrics to template
                **{k: v for k, v in metrics_data.items() if v is not None} 
            }

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

            return render(request, "project1/results.html", context)
            
    return render(request, "project1/train.html")

# Create your views here.
