import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ml_backbone import ml_manager

def index(request):
    """Renders dashboard user interface workspace template context view."""
    return render(request, 'project3/index.html')

def get_metrics(request):
    """Returns baseline stats for Task 1 and Task 2."""
    if request.method == 'GET':
        return JsonResponse({
            'baseline_accuracy': ml_manager.baseline_acc,
            'expert_accuracy': ml_manager.expert_acc,
            'total_test_samples': len(ml_manager.y_test)
        })
    return JsonResponse({'error': 'Method not allowed'}, status=405)

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