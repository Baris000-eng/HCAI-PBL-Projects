import numpy as np
from datasets import load_dataset 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class DeferralSystemManager:
    def __init__(self):

        np.random.seed(42)

        # Import the ag_news data
        dataset = load_dataset("fancyzhx/ag_news")
        self.categories = ['World', 'Sports', 'Business', 'Sci/Tech']

        # Split the imported data into train and test parts 
        self.X_train_raw = dataset['train']['text']
        self.y_train = np.array(dataset['train']['label'])
        self.X_test_raw = dataset['test']['text']
        self.y_test = np.array(dataset['test']['label'])

        # Text Feature Extraction
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.X_train = self.vectorizer.fit_transform(self.X_train_raw).toarray()
        self.X_test = self.vectorizer.transform(self.X_test_raw).toarray()

        # Baseline Classifier
        self.baseline_model = LogisticRegression(C=0.4, max_iter=200)
        self.baseline_model.fit(self.X_train, self.y_train)
        self.baseline_test_preds = self.baseline_model.predict(self.X_test)
        self.baseline_acc = float(accuracy_score(self.y_test, self.baseline_test_preds))

        # Simulated Expert
        self.expert_test_preds = self.simulate_expert_predict(self.y_test)
        self.expert_acc = float(accuracy_score(self.y_test, self.expert_test_preds))

        # Active Learning Setup
        self.AL_pool_indices = list(range(len(self.X_train)))
        print("Number of news in the training dataset: "+str(len(self.AL_pool_indices))+"")
        
        self.AL_labeled_indices = list(np.random.choice(self.AL_pool_indices, size=2000, replace=False))
        for idx in self.AL_labeled_indices:
            if idx in self.AL_pool_indices:
                self.AL_pool_indices.remove(idx)

        # Active learning model training stage 
        self.active_learning_model = LogisticRegression(C=0.4, max_iter=200)
        self.active_learning_model.fit(self.X_train[self.AL_labeled_indices], self.y_train[self.AL_labeled_indices])

    def simulate_expert_predict(self, y_true):
        """Simulates bounded human expert decisions."""
        expert_preds = []
        for label in y_true:
            # Specific domain specialization
            if label == 1:  
                expert_preds.append(label)
            else:
                if np.random.rand() < 0.90:
                    expert_preds.append(label)
                else:
                    remaining_classes = [c for c in range(4) if c != label]
                    expert_preds.append(np.random.choice(remaining_classes))
        return np.array(expert_preds)

    def process_learning_to_defer(self, threshold):
        """Dispatches low-confidence choices to human or expert pipelines."""
        probs = self.baseline_model.predict_proba(self.X_test)
        max_probs = np.max(probs, axis=1)
        classifier_preds = np.argmax(probs, axis=1)

        final_predictions = []
        deferral_count = 0
        undeferral_count = 0
        true_deferral_count = 0 
        false_deferral_count = 0 


        for i in range(len(self.X_test)):
            is_classifier_correct = (classifier_preds[i] == self.y_test[i])
            is_expert_correct = (self.expert_test_preds[i] == self.y_test[i])

            if max_probs[i] < threshold:
                final_predictions.append(self.expert_test_preds[i])
                deferral_count += 1

                if not is_classifier_correct:
                    true_deferral_count+=1
                else:
                    false_deferral_count+=1

            else:
                final_predictions.append(classifier_preds[i])
                undeferral_count += 1

        system_accuracy = float(accuracy_score(self.y_test, final_predictions))
        denom = true_deferral_count + false_deferral_count
        true_deferral_rate = float(true_deferral_count / denom) if denom > 0 else 0.0
        false_deferral_rate = float(false_deferral_count / denom) if denom > 0 else 0.0

        return {
            'system_accuracy': system_accuracy,
            'deferral_rate': float(deferral_count / len(self.X_test)),
            'true_deferral_rate': true_deferral_rate,
            'false_deferral_rate':  false_deferral_rate,
            'deferred_count': deferral_count,
            'undeferred_count': undeferral_count,
            'true_deferred_count': true_deferral_count,
            'false_deferred_count': false_deferral_count,
            'total_count': len(self.X_test)
        }

    def get_next_uncertain_sample(self):
        """Selects data instances with lowest prediction confidence scores."""
        if not self.AL_pool_indices:
            return None

        pool_probs = self.active_learning_model.predict_proba(self.X_train[self.AL_pool_indices])
        max_pool_probs = np.max(pool_probs, axis=1)

        uncertain_pool_idx = np.argmin(max_pool_probs)
        global_idx = self.AL_pool_indices[uncertain_pool_idx]
        
        simulated_label = self.simulate_expert_predict([self.y_train[global_idx]])[0]

        return {
            'sample_index': int(global_idx),
            'text': self.X_train_raw[global_idx],
            'true_label': int(self.y_train[global_idx]),
            'simulated_expert_label': int(simulated_label),
            'categories': self.categories
        }

    def process_query_update(self, idx, chosen_label):
        """Absorbs feedback loop entries and updates the active learner."""
        if idx in self.AL_pool_indices:
            self.AL_pool_indices.remove(idx)
        if idx not in self.AL_labeled_indices:
            self.AL_labeled_indices.append(idx)


        self.y_train[idx] = chosen_label

        # Retrain dynamic tracker model instance
        X_train = self.X_train[self.AL_labeled_indices]
        y_train = self.y_train[self.AL_labeled_indices]
        self.active_learning_model.fit(X_train, y_train)

        al_preds = self.active_learning_model.predict(self.X_test)
        current_al_acc = float(accuracy_score(self.y_test, al_preds))

        return {
            'status': 'success',
            'current_accuracy': current_al_acc,
            'total_labeled_count': len(self.AL_labeled_indices)
        }

# Instantiate singleton class instance across the lifecycle
ml_manager = DeferralSystemManager()