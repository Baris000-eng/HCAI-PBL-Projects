from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def get_machine_learning_model(model_type, final_seed=42, final_weight=None, number_of_epochs=1000):
    """
    Factory method to initialize machine learning models.
    """
    model_type = model_type.lower()
    
    if model_type == 'rfc':
        return RandomForestClassifier(random_state=final_seed, class_weight=final_weight)
    
    elif model_type == 'svm':
        return SVC(random_state=final_seed, class_weight=final_weight, max_iter=number_of_epochs)
    
    elif model_type == 'knn':
        knn_weight = 'distance' if final_weight == 'balanced' else 'uniform'
        return KNeighborsClassifier(weights=knn_weight)
    
    elif model_type == 'dtc':
        return DecisionTreeClassifier(random_state=final_seed, class_weight=final_weight)
    
    elif model_type == 'log_reg':
        return LogisticRegression(random_state=final_seed, class_weight=final_weight, max_iter=number_of_epochs)
    
    elif model_type == 'sgdc':
        return SGDClassifier(random_state=final_seed, class_weight=final_weight, max_iter=number_of_epochs)
    
    else:
        raise ValueError(f"Unsupported model type is detected: {model_type}")

