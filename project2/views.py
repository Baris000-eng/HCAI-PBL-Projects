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

# ============================================================================================
# DATA PIPELINE LOADER & PREPROCESSOR
# ==========================================

def load_and_preprocess_data():
    df = load_penguins().dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    X = df[feature_names].copy()
    y = df['species'].copy()
    master_classes = sorted(list(y.unique()))
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Compute Median Absolute Deviation (MAD) safely
    mads = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0) + 1e-6
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test,
        'X_train_scaled': X_train_scaled, 
        'X_test_scaled': X_test_scaled,
        'scaler': scaler, 
        'mads': mads,
        'feature_names': feature_names, 
        'master_classes': master_classes
    }


# ============================================================================================
# MODEL HYPERPARAMETER SEARCH & OPTIMIZATION 
# ============================================================================================

def optimize_decision_tree(data, lambda_param):
    best_dt_objective = float('inf')  # Looking for the maximum regularized score
    best_dt = None

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    for max_leaves in range(4, 41):
        dtc_clf = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=42)
        dtc_clf.fit(X_train, y_train)

        acc_test = dtc_clf.score(X_test, y_test)
        complexity = dtc_clf.get_n_leaves()

        test_loss = 1 - acc_test
        
        objective = test_loss + (lambda_param * complexity)
        
        if objective < best_dt_objective:
            best_dt_objective = objective
            best_dt = dtc_clf
            
    return best_dt


def optimize_logistic_regression(data, lambda_param):
    best_lr_objective = float('inf')
    best_lr = None

    X_train_scaled = data['X_train_scaled']
    y_train = data['y_train']
    
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    
    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        lr_model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        
        test_acc = lr_model.score(X_test_scaled, y_test)
        sparsity = np.sum(np.abs(lr_model.coef_) > 1e-4)
        
        test_loss_value = 1 - test_acc
        objective = test_loss_value + (lambda_param * sparsity)
        
        if objective < best_lr_objective:
            best_lr_objective = objective
            best_lr = lr_model
            
    return best_lr


def generate_tree_visualization(best_dt, feature_names, master_classes):
    if best_dt is None:
        return ""
    try:
        dot_data = export_graphviz(
            best_dt, out_file=None, feature_names=feature_names,
            class_names=master_classes, filled=True, rounded=True, special_characters=True
        )
        return graphviz.Source(dot_data).pipe(format='svg').decode('utf-8')
    except Exception:
        raw_text = export_text(best_dt, feature_names=feature_names)
        return f'<pre style="text-align: left; font-family: monospace; background: #f8fafc; padding: 15px; border-radius: 6px; border: 1px solid #cbd5e0; line-height: 1.6;">{raw_text}</pre>'


# ===========================================================================================================
# LOCAL & GLOBAL INTERPRETABILITY 
# ===========================================================================================================

def compute_counterfactuals(active_model, data, sample_idx, target_label, model_type):
    X_train = data['X_train']
    
    if sample_idx >= len(X_train) or sample_idx < 0:
        sample_idx = 0
    x_base = X_train.iloc[sample_idx]
    
    N = 4000
    stds = X_train.std().values
    random_noise = np.random.normal(0, 2.0, size=(N, len(data['feature_names']))) * stds
    synthetic_samples = pd.DataFrame(x_base.values + random_noise, columns=data['feature_names'])
    
    if model_type == 'lr':
        synthetic_preds = active_model.predict(data['scaler'].transform(synthetic_samples))
    else:
        synthetic_preds = active_model.predict(synthetic_samples)
        
    valid_indices = np.where(synthetic_preds == target_label)[0]
    
    counterfactuals_list = list()
    if len(valid_indices) > 0:
        valid_samples = synthetic_samples.iloc[valid_indices].copy()
        distances = np.sum(np.abs(valid_samples.values - x_base.values) / data['mads'], axis=1)
        valid_samples['distance'] = distances
        
        best_cf = valid_samples.sort_values(by='distance').head(3)
        counterfactuals_list = best_cf[data['feature_names']].round(2).values.tolist()
        
    return x_base, counterfactuals_list


def compute_pdp_curves(active_model, data, model_type, selected_feature):
    X_train = data['X_train']
    master_classes = data['master_classes']
    model_class_to_idx = {cls: idx for idx, cls in enumerate(active_model.classes_)}
    
    feature_values = X_train[selected_feature].values
    grid = np.linspace(feature_values.min(), feature_values.max(), 20)
    pdp_curves = {cls: [] for cls in master_classes}
    
    for val in grid:
        X_temp = X_train.copy()
        X_temp[selected_feature] = val
        probs = active_model.predict_proba(data['scaler'].transform(X_temp) if model_type == 'lr' else X_temp)
        mean_probs = np.mean(probs, axis=0)
        
        for cls in master_classes:
            if cls in model_class_to_idx:
                pdp_curves[cls].append(float(mean_probs[model_class_to_idx[cls]]))
            else:
                pdp_curves[cls].append(0.0)
                
    return grid, pdp_curves


def compute_ale_curves(active_model, data, model_type, selected_feature):
    X_train = data['X_train']
    master_classes = data['master_classes']
    raw_feature_values = X_train[selected_feature].values
    
    # Create bins and bin centers
    bins = np.percentile(raw_feature_values, np.linspace(0, 100, 41))
    bins = np.unique(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    feat_idx = data['feature_names'].index(selected_feature)
    
    # Initialize a dictionary to hold the differences for each interval
    num_intervals = len(bins) - 1
    local_diffs = {str(cls): np.zeros(num_intervals) for cls in master_classes}
    
    # Create a string-mapped index lookup to prevent silent class mismatch bugs
    model_classes_list = [str(c) for c in active_model.classes_]
    
    # Compute local differences across bins
    for b_idx in range(num_intervals):
        left, right = bins[b_idx], bins[b_idx + 1]
        indices = np.where((raw_feature_values >= left) & (raw_feature_values <= right))[0]
        
        if len(indices) == 0:
            continue
            
        if model_type == 'lr':
            X_bin_raw = X_train.iloc[indices].values
            X_l_raw, X_r_raw = X_bin_raw.copy(), X_bin_raw.copy()
            X_l_raw[:, feat_idx] = left
            X_r_raw[:, feat_idx] = right
            p_left = active_model.predict_proba(data['scaler'].transform(X_l_raw))
            p_right = active_model.predict_proba(data['scaler'].transform(X_r_raw))
        else:
            X_bin = X_train.iloc[indices].copy()
            X_left, X_right = X_bin.copy(), X_bin.copy()
            X_left[selected_feature] = left
            X_right[selected_feature] = right
            p_left = active_model.predict_proba(X_left)
            p_right = active_model.predict_proba(X_right)
            
        mean_diffs = np.mean(p_right - p_left, axis=0)
        
        # Match class predictions reliably
        for idx, model_cls in enumerate(model_classes_list):
            cls_str = str(model_cls)
            if cls_str in local_diffs:
                local_diffs[cls_str][b_idx] = mean_diffs[idx]
                
    # Accumulate, Interpolate to Centers, and Mean-Center the curves
    ale_curves = {}
    for cls in master_classes:
        # Start integration at 0 for the first bin boundary
        accumulated_at_boundaries = np.concatenate(([0], np.cumsum(local_diffs[cls])))
        
        # Linearly interpolate the accumulated effects from boundaries to bin centers
        ale_at_centers = (accumulated_at_boundaries[:-1] + accumulated_at_boundaries[1:]) / 2
        
        # Mean-center the ALE curve so its expected value relative to the data is 0
        # (This uses the count of points per bin to weight the mean properly)
        bin_counts = np.array([
            len(np.where((raw_feature_values >= bins[i]) & (raw_feature_values <= bins[i+1]))[0])
            for i in range(num_intervals)
        ])
        
        # Fallback to simple mean if bin_counts sum to 0
        total_points = np.sum(bin_counts)
        if total_points > 0:
            mean_value = np.sum(ale_at_centers * bin_counts) / total_points
        else:
            mean_value = np.mean(ale_at_centers)
            
        mean_centered = ale_at_centers - mean_value
        ale_curves[cls] = [float(x) for x in mean_centered]
        
    return bin_centers, ale_curves


# ==============================================================================================================================
# PRIMARY VIEW 
# ==============================================================================================================================

def index(request):
    # Build training environment pipelines
    data = load_and_preprocess_data()
    
    # Extract state selectors from the user request
    lambda_param = float(request.GET.get('lambda_value', 0.0))
    model_type = request.GET.get('model_type', 'tree')
    selected_feature = request.GET.get('feature', 'bill_length_mm')
    sample_idx = int(request.GET.get('sample_idx', 0))
    target_label = request.GET.get('target_label', 'Gentoo')
    
    # Train models via structural optimization searches
    best_dt = optimize_decision_tree(data, lambda_param)
    best_lr = optimize_logistic_regression(data, lambda_param)
    
    # Extract metrics from the active model layout
    active_model = best_dt if model_type == 'tree' else best_lr
    if model_type == 'tree':
        active_accuracy = active_model.score(data['X_test'], data['y_test'])
        active_complexity = active_model.get_n_leaves()
    else:
        active_accuracy = active_model.score(data['X_test_scaled'], data['y_test'])
        active_complexity = int(np.sum(np.abs(active_model.coef_) > 1e-4))
        
    # Extract structural and behavioral configurations
    tree_graph_html = generate_tree_visualization(best_dt, data['feature_names'], data['master_classes'])
    x_base, counterfactuals_list = compute_counterfactuals(active_model, data, sample_idx, target_label, model_type)
    grid, pdp_curves = compute_pdp_curves(active_model, data, model_type, selected_feature)
    bin_centers, ale_curves = compute_ale_curves(active_model, data, model_type, selected_feature)
    
    # Send the context to the browser template 
    context = {
        'lambda_value': lambda_param,
        'model_type': model_type,
        'selected_feature': selected_feature,
        'feature_names': data['feature_names'],
        'accuracy': round(active_accuracy, 4),
        'complexity_metric': active_complexity,
        'tree_graph': tree_graph_html,
        'grid_labels': json.dumps([round(x, 2) for x in grid]),
        'pdp_curves': json.dumps(pdp_curves),
        'bin_centers': json.dumps([round(x, 2) for x in bin_centers]),
        'ale_curves': json.dumps(ale_curves),
        'base_sample': x_base.round(2).to_dict(),
        'counterfactuals_list': counterfactuals_list,
        'target_label': target_label,
        'classes': data['master_classes']
    }
    return render(request, 'project2/index.html', context)

# Create your views here.
