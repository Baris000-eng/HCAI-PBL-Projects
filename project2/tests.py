from django.test import TestCase
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from .load_and_preprocess_data import load_and_preprocess_data
from .hyperparameter_search_and_optimization import optimize_decision_tree, optimize_logistic_regression, generate_tree_visualization

class DataPreprocessingTestCase(TestCase):
    def setUp(self):
        self.data = load_and_preprocess_data()

    def test_load_and_preprocess_data_keys(self):
        expected_keys = [
            'X_train', 'X_test', 'y_train', 'y_test', 
            'X_train_scaled', 'X_test_scaled', 'scaler', 
            'mads', 'feature_names', 'master_classes'
        ]
        for key in expected_keys:
            self.assertIn(key, self.data)

    def test_data_splitting(self):
        total_len = len(self.data['X_train']) + len(self.data['X_test'])
        test_ratio = len(self.data['X_test']) / total_len
        self.assertAlmostEqual(test_ratio, 0.3, delta=0.05)

    def test_scaling(self):
        scaled_data = self.data['X_train_scaled']
        means = np.mean(scaled_data, axis=0)
        stds = np.std(scaled_data, axis=0)
        
        np.testing.assert_allclose(means, 0, atol=1e-7)
        np.testing.assert_allclose(stds, 1, atol=1e-7)

    def test_mads_shape(self):
        self.assertEqual(len(self.data['mads']), len(self.data['feature_names']))


class OptimizationTestCase(TestCase):
    def setUp(self):
        self.data = load_and_preprocess_data()

    def test_optimize_decision_tree(self):
        best_dt = optimize_decision_tree(self.data, lambda_param=0.01)
        self.assertIsNotNone(best_dt)
        self.assertIsInstance(best_dt, DecisionTreeClassifier)

    def test_optimize_logistic_regression(self):
        best_lr = optimize_logistic_regression(
            self.data, lambda_param=0.01, penalty='l2', solver='lbfgs'
        )
        self.assertIsNotNone(best_lr)
        self.assertIsInstance(best_lr, LogisticRegression)

    def test_generate_tree_visualization(self):
        best_dt = optimize_decision_tree(self.data, lambda_param=0.01)
        
        viz = generate_tree_visualization(
            best_dt, self.data['feature_names'], self.data['master_classes']
        )
        
        is_svg = "<svg" in viz 
        is_pre = "<pre" in viz 
        
        self.assertTrue(
              is_svg or is_pre, 
            f"Beklenen formatta değil! Gelen değer: {viz}"
        )

    def test_generate_tree_visualization_none(self):
        viz = generate_tree_visualization(None, None, None)
        self.assertEqual(viz, "")