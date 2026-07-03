import os 
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ml_backbone import ml_manager

@csrf_exempt
def clear_plot(request, save_path="accuracy_growth_graph.png"): 
    """Clears the saved plot file from the user interface and static directory"""
    try:
        if os.path.exists(save_path): 
            os.remove(save_path)
        return JsonResponse({"status": "success", "message": "Graph is deleted from the server."})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
def trigger_plot_metrics(request):
    if request.method == 'POST':
        try:
            current_file_path = os.path.abspath(__file__)
            app_directory = os.path.dirname(current_file_path)
            base_directory = os.path.dirname(app_directory)

            # Target project's static folder to save the plot image 
            save_path = os.path.join(base_directory, 'static', 'accuracy_growth_graph.png')
            
            # Ensure static directory exists
            output_dir = os.path.dirname(save_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            image_data_uri = ml_manager.plot_metrics_and_show_in_web(save_path=save_path)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Plot executed successfully.',
                'image_data': image_data_uri  
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

def index(request):
    """Renders dashboard user interface workspace template context view."""
    recent_al_accuracy = ml_manager.accuracy_history[-1]
    recent_total_queries = ml_manager.query_history[-1]
    manual_queries = recent_total_queries - 2000
    active_learning_context = {
        'recent_al_accuracy': recent_al_accuracy * 100,
        'recent_query_count': recent_total_queries,
        'manual_query_count': manual_queries
    }
    return render(request, 'project3/index.html', active_learning_context)

def get_metrics(request):
    """Returns baseline statistics."""
    if request.method == 'GET':
        return JsonResponse({
            'baseline_accuracy': ml_manager.baseline_acc,
            'expert_accuracy': ml_manager.expert_acc,
            'total_test_samples': len(ml_manager.y_test)
        })
    return JsonResponse({'error': 'Method is not allowed'}, status=405)

@csrf_exempt
def learning_to_defer(request):
    """Evaluates combined operational system accuracies based on variable confidence thresholds."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body) if request.body else {}
            threshold = float(data.get('threshold', 0.75))
            
            results = ml_manager.process_learning_to_defer(threshold)
            return JsonResponse(results)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def al_next_sample(request):
    """Selects next active learning candidate matching uncertainty strategies."""
    if request.method == 'GET':
        sample_payload = ml_manager.get_next_uncertain_sample()
        if not sample_payload:
            return JsonResponse({'error': 'No remaining samples in pool'}, status=400)
        return JsonResponse(sample_payload)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def al_query(request):
    """Integrates human annotation inputs to update loop parameters."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body) if request.body else {}
            idx = int(data.get('sample_index'))
            chosen_label = int(data.get('label'))
            
            update_results = ml_manager.process_query_update(idx, chosen_label)
            return JsonResponse(update_results)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

