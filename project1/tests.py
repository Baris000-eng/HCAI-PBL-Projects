from django.test import TestCase, Client
from django.urls import reverse
import pandas as pd

class ModelTrainingTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        # Create a sample dataframe and save it to the session to simulate an upload
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        session = self.client.session
        session['df_json'] = df.to_json()
        session.save()

    def test_train_view_success_with_metrics(self):
        """Test that the training view calculates requested metrics."""
        url = reverse('project1:train')
        data = {
            'model_type': 'rfc',
            'split_ratio': 0.2,
            'eval_metrics': ['accuracy', 'precision', 'f1_score'],
            'random_state': 42,
            'class_weight': 'none',
            'iterations': 100
        }
        
        response = self.client.post(url, data)
        
        self.assertEqual(response.status_code, 200)
        # Verify requested metrics exist in context
        self.assertIn('accuracy', response.context)
        self.assertIn('precision', response.context)
        self.assertIn('f1_score', response.context)

    def test_metrics_not_requested_are_absent(self):
        """Test that unrequested metrics do not appear in the context."""
        url = reverse('project1:train')
        
        # Only request accuracy
        data = {
            'model_type': 'rfc',
            'split_ratio': 0.2,
            'eval_metrics': ['accuracy'], 
            'random_state': 42,
            'class_weight': 'none',
            'iterations': 100
        }
        
        response = self.client.post(url, data)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify requested metric exists
        self.assertIn('accuracy', response.context)
        
        # Verify unrequested metrics do not exist
        self.assertNotIn('f1_score', response.context)
        self.assertNotIn('confusion_matrix', response.context)
        self.assertNotIn('recall', response.context)

    def test_invalid_model_type(self):
        """Test that an invalid model type triggers an error."""
        url = reverse('project1:train')
        data = {
            'model_type': 'not_a_real_model',
            'split_ratio': 0.2,
        }
        
        # This expects your factory to raise a ValueError
        with self.assertRaises(ValueError):
            self.client.post(url, data)

    def test_no_session_data_redirect(self):
        """Test behavior when there is no data in the session."""
        # Clear the session
        session = self.client.session
        session.clear()
        session.save()
        
        url = reverse('project1:train')
        response = self.client.post(url, {})
        
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'project1/train.html')