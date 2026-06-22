import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CATEGORIES = ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast']

class DeferralSystemManager:
    def __init__(self):
        print("Initializing Machine Learning Datasets & Baselines...")
        # Simulating 4-class AG News setup using 20Newsgroups slice
        self.newsgroups_train = fetch_20newsgroups(subset='train', categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))
        self.newsgroups_test = fetch_20newsgroups(subset='test', categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))

        self.X_train_raw = self.newsgroups_train.data
        self.y_train = self.newsgroups_train.target
        self.X_test_raw = self.newsgroups_test.data
        self.y_test = self.newsgroups_test.target

        # Text Feature Extraction
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.X_train = self.vectorizer.fit_transform(self.X_train_raw).toarray()
        self.X_test = self.vectorizer.transform(self.X_test_raw).toarray()

        # Task 1: Baseline Classifier
        self.baseline_model = LogisticRegression(C=1.0)
        self.baseline_model.fit(self.X_train, self.y_train)
        self.baseline_preds = self.baseline_model.predict(self.X_test)
        self.baseline_acc = float(accuracy_score(self.y_test, self.baseline_preds))

        # Task 2: Simulated Expert
        self.expert_test_preds = self.simulate_expert_predict(self.X_test_raw, self.y_test)
        self.expert_acc = float(accuracy_score(self.y_test, self.expert_test_preds))

        # Task 4: Active Learning Setup
        self.AL_pool_indices = list(range(len(self.X_train)))
        self.AL_queried_indices = []
        
        np.random.seed(42)
        self.AL_labeled_indices = list(np.random.choice(self.AL_pool_indices, size=20, replace=False))
        for idx in self.AL_labeled_indices:
            if idx in self.AL_pool_indices:
                self.AL_pool_indices.remove(idx)

        # Dynamic model tracking state
        self.al_model = LogisticRegression(C=1.0)
        self.al_model.fit(self.X_train[self.AL_labeled_indices], self.y_train[self.AL_labeled_indices])

    def simulate_expert_predict(self, X_raw, y_true):
        """Simulates bounded human expert decisions."""
        expert_preds = []
        for label in y_true:
            if label == 1:  # Niche domain specialization
                expert_preds.append(label)
            else:
                if np.random.rand() < 0.65:
                    expert_preds.append(label)
                else:
                    remaining_classes = [c for c in range(4) if c != label]
                    expert_preds.append(np.random.choice(remaining_classes))
        return np.array(expert_preds)

    def process_learning_to_defer(self, threshold):
        """Task 3: Dispatches low-confidence choices to human or expert pipelines."""
        probs = self.baseline_model.predict_proba(self.X_test)
        max_probs = np.max(probs, axis=1)
        classifier_preds = np.argmax(probs, axis=1)

        final_predictions = []
        deferral_count = 0

        for i in range(len(self.X_test)):
            if max_probs[i] < threshold:
                final_predictions.append(self.expert_test_preds[i])
                deferral_count += 1
            else:
                final_predictions.append(classifier_preds[i])

        system_accuracy = float(accuracy_score(self.y_test, final_predictions))
        return {
            'system_accuracy': system_accuracy,
            'deferral_rate': float(deferral_count / len(self.X_test)),
            'deferred_count': deferral_count,
            'total_count': len(self.X_test)
        }

    def get_next_uncertain_sample(self):
        """Task 4: Selects data instances with lowest prediction confidence scores."""
        if not self.AL_pool_indices:
            return None

        pool_probs = self.al_model.predict_proba(self.X_train[self.AL_pool_indices])
        max_pool_probs = np.max(pool_probs, axis=1)

        uncertain_pool_idx = np.argmin(max_pool_probs)
        global_idx = self.AL_pool_indices[uncertain_pool_idx]
        
        simulated_label = self.simulate_expert_predict([self.X_train_raw[global_idx]], [self.y_train[global_idx]])[0]

        return {
            'sample_index': int(global_idx),
            'text': self.X_train_raw[global_idx],
            'true_label': int(self.y_train[global_idx]),
            'simulated_expert_label': int(simulated_label),
            'categories': CATEGORIES
        }

    def process_query_update(self, idx, chosen_label):
        """Tasks 4 & 5: Absorbs feedback loop entries and updates the active learner."""
        if idx in self.AL_pool_indices:
            self.AL_pool_indices.remove(idx)
        if idx not in self.AL_labeled_indices:
            self.AL_labeled_indices.append(idx)


        self.y_train[idx] = chosen_label

        # Retrain dynamic tracker model instance
        self.al_model = LogisticRegression(C=1.0)
        X_train = self.X_train[self.AL_labeled_indices]
        y_train = self.y_train[self.AL_labeled_indices]
        self.al_model.fit(X_train, y_train)

        al_preds = self.al_model.predict(self.X_test)
        current_al_acc = float(accuracy_score(self.y_test, al_preds))

        return {
            'status': 'success',
            'current_accuracy': current_al_acc,
            'total_labeled_count': len(self.AL_labeled_indices)
        }

# Instantiate singleton class instance across the lifecycle
ml_manager = DeferralSystemManager()