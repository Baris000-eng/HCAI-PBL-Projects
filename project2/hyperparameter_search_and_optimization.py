import numpy as np 
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.linear_model import LogisticRegression



def optimize_decision_tree(data, lambda_param):
    best_dt_objective = float('inf')  
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


def optimize_logistic_regression(data, lambda_param, penalty, solver):
    best_lr_objective = float('inf')
    best_lr = None

    X_train_scaled = data['X_train_scaled']
    y_train = data['y_train']
    
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    
    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:

        lr_model = LogisticRegression(C=c_val, max_iter=1000, penalty=penalty, solver=solver,random_state=42)
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
            decision_tree=best_dt, 
            out_file=None, 
            feature_names=feature_names,
            class_names=master_classes, 
            filled=True, 
            rounded=True, 
            special_characters=True
        )
        return graphviz.Source(dot_data).pipe(format='svg').decode('utf-8')
    except Exception:
        raw_text = export_text(best_dt, feature_names=feature_names)
        return f'<pre style="text-align: left; font-family: monospace; background: #f8fafc; padding: 15px; border-radius: 6px; border: 1px solid #cbd5e0; line-height: 1.6;">{raw_text}</pre>'
    