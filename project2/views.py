import json
import numpy as np
import pandas as pd
import graphviz
from django.shortcuts import render
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def index(request):
    # --- Data Loading and Preprocessing ---
    df = load_penguins().dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    X = df[feature_names].copy()
    y = df['species'].copy()
    
    master_classes = sorted(list(y.unique())) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Calculate Median Absolute Deviation for distance weighting in counterfactual search 
    mads = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0) + 1e-6

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Obtain User-Selected Inputs 
    lambda_param = float(request.GET.get('lambda_value', 0.0))
    model_type = request.GET.get('model_type', 'tree')
    selected_feature = request.GET.get('feature', 'bill_length_mm')
    
    sample_idx = int(request.GET.get('sample_idx', 0))
    target_label = request.GET.get('target_label', 'Gentoo')

    # --- Model Fitting & Structural Hyperparameter Search ---
    
    # Decision Tree Optimization: Minimize (1 - Accuracy) + Lambda * Complexity
    best_dt_objective = float('inf')
    best_dt = None
    for max_leaves in range(2, 151):
        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=42)
        clf.fit(X_train, y_train)
        
        acc_test = clf.score(X_test, y_test)
        complexity = clf.get_n_leaves()
        
        test_error = 1.0 - acc_test
        objective = test_error + (lambda_param * complexity)
        
        if objective < best_dt_objective:
            best_dt_objective = objective
            best_dt = clf

    # Logistic Regression Optimization: Minimize (1 - Accuracy) + Lambda * Sparsity
    best_lr_objective = float('inf')
    best_lr = None
    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        acc_test = model.score(X_test_scaled, y_test)
        sparsity = np.sum(np.abs(model.coef_) > 1e-4) 
        
        test_error = 1.0 - acc_test
        objective = test_error + (lambda_param * sparsity)
        
        if objective < best_lr_objective:
            best_lr_objective = objective
            best_lr = model

    # Select operational model architecture
    active_model = best_dt if model_type == 'tree' else best_lr
    
    # Map layout indexes of the active model back to master classes
    model_class_to_idx = {cls: idx for idx, cls in enumerate(active_model.classes_)}
    
    # Extract structural metrics for dashboard display
    active_accuracy = active_model.score(X_test, y_test) if model_type == 'tree' else active_model.score(X_test_scaled, y_test)
    active_leaves_or_sparsity = active_model.get_n_leaves() if model_type == 'tree' else int(np.sum(np.abs(active_model.coef_) > 1e-4))

    # Dynamic Tree Generation
    tree_graph_html = ""
    if model_type == 'tree' and best_dt is not None:
        try:
            dot_data = export_graphviz(
                best_dt, out_file=None, feature_names=feature_names,
                class_names=master_classes, filled=True, rounded=True, special_characters=True
            )
            graph = graphviz.Source(dot_data)
            tree_graph_html = graph.pipe(format='svg').decode('utf-8')
        except Exception:
            raw_text = export_text(best_dt, feature_names=feature_names)
            tree_graph_html = f'<pre style="text-align: left; font-family: monospace; background: #f8fafc; padding: 15px; border-radius: 6px; border: 1px solid #cbd5e0; line-height: 1.6;">{raw_text}</pre>'

    # Boundary control to avoid sample index overflow errors
    if sample_idx >= len(X_train) or sample_idx < 0:
        sample_idx = 0
    x_base = X_train.iloc[sample_idx]
    
    # Local random perturbation sampling
    N = 3500
    stds = X_train.std().values
    random_noise = np.random.normal(0, 3.0, size=(N, len(feature_names))) * stds
    synthetic_samples = pd.DataFrame(x_base.values + random_noise, columns=feature_names)
    
    if model_type == 'lr':
        synthetic_preds = active_model.predict(scaler.transform(synthetic_samples))
    else:
        synthetic_preds = active_model.predict(synthetic_samples)
        
    valid_indices = np.where(synthetic_preds == target_label)[0]
    
    counterfactuals_list = []
    if len(valid_indices) > 0:
        valid_samples = synthetic_samples.iloc[valid_indices].copy()
        distances = np.sum(np.abs(valid_samples.values - x_base.values) / mads, axis=1)
        valid_samples['distance'] = distances
        
        best_cf = valid_samples.sort_values(by='distance').head(3)
        counterfactuals_list = best_cf[feature_names].round(2).values.tolist()

    # Manual Computation of PDP 
    feature_values = X_train[selected_feature].values
    grid = np.linspace(feature_values.min(), feature_values.max(), 20)
    
    # Initialize dictionary explicitly for all three master classes
    pdp_curves = {cls: [] for cls in master_classes}
    
    for val in grid:
        X_temp = X_train.copy()
        X_temp[selected_feature] = val
        probs = active_model.predict_proba(scaler.transform(X_temp) if model_type == 'lr' else X_temp)
        mean_probs = np.mean(probs, axis=0)
        
        for cls in master_classes:
            if cls in model_class_to_idx:
                actual_idx = model_class_to_idx[cls]
                pdp_curves[cls].append(float(mean_probs[actual_idx]))
            else:
                pdp_curves[cls].append(0.0) # Fallback baseline if model drops a class entirely

    #  Manual Computation of ALE 
    raw_feature_values = X_train[selected_feature].values
    
    # Intervals provides higher precision to catch local decision bounds
    bins = np.percentile(raw_feature_values, np.linspace(0, 100, 41)) 
    bins = np.unique(bins) 
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize step array explicitly for all three master classes
    ale_accumulated = {cls: np.zeros(len(bin_centers)) for cls in master_classes}
    
    for b_idx in range(0, len(bins)-1):
        left, right = bins[b_idx], bins[b_idx+1]
        indices = np.where((raw_feature_values >= left) & (raw_feature_values <= right))[0]
        
        if len(indices) == 0:
            continue
            
        X_bin = X_train.iloc[indices].copy()
        X_left, X_right = X_bin.copy(), X_bin.copy()
        
        X_left[selected_feature] = left
        X_right[selected_feature] = right
        
        if model_type == 'lr':
            p_left = active_model.predict_proba(scaler.transform(X_left))
            p_right = active_model.predict_proba(scaler.transform(X_right))
        else:
            p_left = active_model.predict_proba(X_left)
            p_right = active_model.predict_proba(X_right)
        
        mean_diff = np.mean(p_right - p_left, axis=0)
        
        for cls in master_classes:
            if cls in model_class_to_idx:
                actual_idx = model_class_to_idx[cls]
                ale_accumulated[cls][b_idx] = mean_diff[actual_idx]
            else:
                ale_accumulated[cls][b_idx] = 0.0

    display_bin_centers = [round(x, 2) for x in bin_centers]
    ale_curves = {cls: [] for cls in master_classes}
    
    for cls in master_classes:
        cumulative = np.cumsum(ale_accumulated[cls])
        # Accurate mean-centering adjustment over the calculated step progression
        if len(cumulative) > 0:
            cumulative = cumulative - np.mean(cumulative)
        ale_curves[cls] = [float(x) for x in cumulative]

    # --- Render Context Payload ---
    context = {
        'lambda_value': lambda_param,
        'model_type': model_type,
        'selected_feature': selected_feature,
        'feature_names': feature_names,
        'accuracy': round(active_accuracy, 4),
        'complexity_metric': active_leaves_or_sparsity,
        'tree_graph': tree_graph_html,
        'grid_labels': json.dumps([round(x, 2) for x in grid]),
        'pdp_curves': json.dumps(pdp_curves),
        'bin_centers': json.dumps(display_bin_centers),
        'ale_curves': json.dumps(ale_curves),
        'base_sample': x_base.round(2).to_dict(),
        'counterfactuals_list': counterfactuals_list,
        'target_label': target_label,
        'classes': master_classes # Ensure frontend selects dropdowns use master list
    }
    return render(request, 'project2/index.html', context)

# Create your views here.
