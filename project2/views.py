import json
import numpy as np
from django.shortcuts import render


from .load_and_preprocess_data import load_and_preprocess_data
from .hyperparameter_search_and_optimization import optimize_decision_tree, optimize_logistic_regression, generate_tree_visualization
from .local_and_global_interpretability import compute_counterfactuals, compute_pdp_curves, compute_ale_curves
from .models import PenguinObservation

def index(request):
   
    data = load_and_preprocess_data()
    
    # Extract the values of the parameters selected by the user in the user interface
    lambda_param = float(request.GET.get('lambda_value', 0.0))
    model_type = request.GET.get('model_type', 'dtrc')
    selected_feature = request.GET.get('feature', 'bill_length_mm')
    sample_idx = int(request.GET.get('sample_idx', 0))
    target_label = request.GET.get('target_label', 'Gentoo')
    weight_solver = request.GET.get('weight_solver', 'lbfgs')
    regularizer_penalty = request.GET.get('regularizer_penalty', 'l2')
    
    # Train models via structural optimization searches
    best_dt = optimize_decision_tree(data, lambda_param)  
    best_lr = optimize_logistic_regression(data, lambda_param, regularizer_penalty, weight_solver)

    
    # Extract metrics from the active model layout
    if model_type == 'dtrc':
        active_model = best_dt
        active_accuracy = active_model.score(data['X_test'], data['y_test'])
        active_complexity_measure = active_model.get_n_leaves()
    elif model_type == 'lr':
        active_model = best_lr
        active_accuracy = active_model.score(data['X_test_scaled'], data['y_test'])
        active_complexity_measure = int(np.sum(np.abs(active_model.coef_) > 1e-4))
    else: 
        raise ValueError(f"Unsupported model type is detected: {model_type}")
    
    # 1. Fetch the specific observation from your Django Database model
    try:
        db_sample = PenguinObservation.objects.all()[sample_idx]
    except IndexError:
        # Fallback security if sample_idx goes out of range
        db_sample = PenguinObservation.objects.first()
        
    
    # Generate structural and behavioral data for visualizations and interpretability analyses 

    # Get the best decision tree visual
    tree_graph_html = generate_tree_visualization(best_dt, data['feature_names'], data['master_classes'])

    # Get the best 5 counterfactuals 
    x_base, counterfactuals_list = compute_counterfactuals(
        active_model=active_model, 
        data=data, 
        sample_idx=sample_idx, 
        target_label=target_label, 
        model_type=model_type, 
        k=5
    )

    # Show the PDP and ALE curve for the selected feature 
    grid, pdp_curves = compute_pdp_curves(
        active_model=active_model, 
        data=data, 
        model_type=model_type, 
        selected_feature=selected_feature
    )
    bin_centers, ale_curves = compute_ale_curves(
        active_model=active_model, 
        data=data, 
        model_type=model_type, 
        selected_feature=selected_feature
    )
    
    # Send the context to the browser template 
    context = {
        'lambda_value': lambda_param,
        'model_type': model_type,
        'selected_feature': selected_feature,
        'feature_names': data['feature_names'],
        'accuracy': round(active_accuracy, 6),
        'complexity_metric': active_complexity_measure,
        'tree_graph': tree_graph_html,
        'grid_labels': json.dumps([round(x, 2) for x in grid]),
        'pdp_curves': json.dumps(pdp_curves),
        'bin_centers': json.dumps([round(x, 2) for x in bin_centers]),
        'ale_curves': json.dumps(ale_curves),
        'base_sample': x_base.round(2).to_dict(),
        'counterfactuals_list': counterfactuals_list,
        'target_label': target_label,
        'sample_idx': sample_idx,
        'weight_solver': weight_solver,
        'regularizer_penalty': regularizer_penalty,
        'classes': data['master_classes'],
        'db_sample_id': db_sample.id
    }
    return render(request, 'project2/index.html', context)



# Create your views here.
