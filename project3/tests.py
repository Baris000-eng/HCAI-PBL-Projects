import numpy as np
from unittest.mock import patch
from django.test import TestCase
from .ml_backbone import DeferralSystemManager 

class TestDeferralSystemManager(TestCase):

    @patch('project3.ml_backbone.load_dataset')
    def setUp(self, mock_load_dataset):
        """Sets up a lightweight mock dataset to prevent slow downloading during testing."""
        # Create fake text and labels for training (6 samples) and testing (4 samples)
        fake_dataset = {
            'train': {
                'text': [
                    "Oil prices surge amid global supply concerns.", 
                    "Local team wins the championship in overtime.",
                    "Tech company launches a new AI smartphone.",
                    "Stocks plummet following federal rate updates.",
                    "Quarterback signs a record-breaking contract extension.",
                    "New software patch fixes critical kernel vulnerability."
                ],
                'label': [2, 1, 3, 2, 1, 3] # Business=2, Sports=1, Sci/Tech=3
            },
            'test': {
                'text': [
                    "Market rates stabilize after volatile trading session.",
                    "Forward scores hat-trick in seasonal opener.",
                    "Scientists discover a new exoplanet using AI.",
                    "Global leaders meet to discuss treaty terms."
                ],
                'label': [2, 1, 3, 0] # 0 = World
            }
        }
        mock_load_dataset.return_value = fake_dataset

        # Define a smart mock for random choices to handle both scalars and size arrays safely
        def smart_random_choice(a, size=None, replace=True, p=None):
            if size is not None:
                # Active learning pool setup: return clean indices capped by available training size
                limit = a if isinstance(a, int) else len(a)
                return np.array([i % limit for i in range(size)])
            else:
                # Expert prediction simulations: return a single scalar value
                if isinstance(a, int):
                    return 0
                return a[0] if len(a) > 0 else 0

        # Patch the random choice inside __init__ using our smart side-effect
        with patch('numpy.random.choice', side_effect=smart_random_choice):
            self.manager = DeferralSystemManager()

    def test_initialization(self):
        """Verifies that datasets, vectorizer, and baseline models initialize with correct shapes."""
        self.assertEqual(len(self.manager.X_train_raw), 6)
        self.assertEqual(len(self.manager.X_test_raw), 4)
        self.assertEqual(len(self.manager.categories), 4)
        
        # Verify vectorizer capped at max_features (or smaller if dictionary is small)
        self.assertTrue(self.manager.X_train.shape[1] <= 5000)
        self.assertEqual(self.manager.X_test.shape[0], 4)

    def test_simulate_expert_predict(self):
        """Ensures the expert simulation always respects the strict matrix size bounds."""
        test_labels = np.array([0, 1, 2, 3])
        expert_predictions = self.manager.simulate_expert_predict(test_labels)
        
        self.assertEqual(len(expert_predictions), 4)
        # Category '1' (Sports) must be predicted perfectly due to expert specialization code rule
        self.assertEqual(expert_predictions[1], 1)

    def test_process_learning_to_defer(self):
        """Validates system routing metrics when setting thresholds."""
        # Low threshold: Classifier handles everything
        results_low = self.manager.process_learning_to_defer(threshold=0.0)
        self.assertEqual(results_low['deferred_count'], 0)
        self.assertEqual(results_low['total_count'], 4)

        # High threshold: Force system to route predictions toward the expert pipeline
        results_high = self.manager.process_learning_to_defer(threshold=1.0)
        self.assertEqual(results_high['undeferred_count'], 0)
        self.assertEqual(results_high['deferred_count'], 4)
        
        # Structure validations
        self.assertIn('system_accuracy', results_high)
        self.assertIn('deferral_rate', results_high)

    def test_get_next_uncertain_sample(self):
        """Validates retrieval of samples with the lowest prediction confidence scores from the pool."""
        sample_data = self.manager.get_next_uncertain_sample()
        
        if sample_data:
            self.assertIn('sample_index', sample_data)
            self.assertIn('text', sample_data)
            self.assertIn('true_label', sample_data)
            self.assertIn('simulated_expert_label', sample_data)
            self.assertEqual(sample_data['categories'], self.manager.categories)

    def test_process_query_update(self):
        """Tests that processing a user/expert feedback loop successfully updates the active learning cycle."""
        # Get an active index out of the remaining pool
        self.manager.AL_pool_indices = [3, 4, 5]
        target_idx = 3
        chosen_label = 2
        
        updated_metrics = self.manager.process_query_update(target_idx, chosen_label)
        
        self.assertNotIn(target_idx, self.manager.AL_pool_indices)
        self.assertIn(target_idx, self.manager.AL_labeled_indices)

        # Validate the types or values of the returned evaluation metrics 
        self.assertEqual(updated_metrics['status'], 'success')
        self.assertIsInstance(updated_metrics['current_test_accuracy'], float)
        self.assertIsInstance(updated_metrics['accuracy_gain_per_query'], float)
        self.assertGreaterEqual(updated_metrics['disagreement_rate'], 0)