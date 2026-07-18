import numpy as np
import pandas as pd


def compute_counterfactuals(active_model, data, sample_idx, target_label, model_type, k):
    X_train = data['X_train']
    
    # To handle inconsistent sample_idx values 
    if sample_idx >= len(X_train) or sample_idx < 0:
        sample_idx = 0
    x_base = X_train.iloc[sample_idx]
    
    N = 4000
    stds = X_train.std().values
    random_noise = np.random.normal(0, 2.0, size=(N, len(data['feature_names']))) * stds
    synthetic_samples = pd.DataFrame(x_base.values + random_noise, columns=data['feature_names'])
    
    if model_type == 'lr':
        scaled_samples = data['scaler'].transform(synthetic_samples)
        synthetic_preds = active_model.predict(scaled_samples)
    else:
        synthetic_preds = active_model.predict(synthetic_samples)
        
    valid_indices = np.where(synthetic_preds == target_label)[0]
    
    counterfactuals_list = list()
    if len(valid_indices) > 0:
        valid_samples = synthetic_samples.iloc[valid_indices].copy()
        distances = np.sum(np.abs(valid_samples.values - x_base.values) / data['mads'], axis=1)
        valid_samples['distance'] = distances
        
        best_cf = valid_samples.sort_values(by='distance').head(k)
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
    
    # Create an index lookup list 
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

            X_l_df = pd.DataFrame(X_l_raw, columns=data['feature_names'])
            X_r_df = pd.DataFrame(X_r_raw, columns=data['feature_names'])

            p_left = active_model.predict_proba(data['scaler'].transform(X_l_df))
            p_right = active_model.predict_proba(data['scaler'].transform(X_r_df))
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
