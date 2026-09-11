import json
import numpy as np
from unittest.mock import patch, MagicMock

from django.test import TestCase, RequestFactory
from django.urls import reverse

from project4 import views 


class RecommendationSystemTests(TestCase):
    
    @classmethod
    def setUpTestData(cls):
        """Set up dummy movie data to isolate tests from local files/datasets."""
        cls.factory = RequestFactory()
        
        # Mock dataset structured similarly to the actual loaded dataset
        cls.mock_movies = [
            {
                'id': 1,
                'title': 'Movie A',
                'year': 2020,
                'genre': 'Action|Thriller',
                'score': 8.0,
                'feature_vector': [0.0, 1.0, 0.53]  
            },
            {
                'id': 2,
                'title': 'Movie B',
                'year': 2018,
                'genre': 'Comedy',
                'score': 6.5,
                'feature_vector': [1.0, 0.0, 0.65]
            },
            {
                'id': 3,
                'title': 'Movie C',
                'year': 2022,
                'genre': 'Action|Comedy',
                'score': 7.2,
                'feature_vector': [1.0, 1.0, 0.79]
            }
        ]
        
        cls.mock_movie_map = {m['id']: m for m in cls.mock_movies}
        cls.features_dim = len(cls.mock_movies[0]['feature_vector'])

    def setUp(self):
        """Patch global module-level variables for each test."""
        
        self.patcher_movies = patch.object(views, 'movies', self.mock_movies)
        self.patcher_map = patch.object(views, 'movie_map', self.mock_movie_map)
        self.patcher_dim = patch.object(views, 'features_dim', self.features_dim)
        
        self.patcher_movies.start()
        self.patcher_map.start()
        self.patcher_dim.start()

    def tearDown(self):
        """Stop patchers after each test run."""
        self.patcher_movies.stop()
        self.patcher_map.stop()
        self.patcher_dim.stop()

    # --- Pure Mathematical & Helper Functions Tests ---

    def test_bradley_terry_update(self):
        """Verify that Bradley-Terry updates weights in the direction of the chosen item."""
        w = np.zeros(3)
        x_a = np.array([1.0, 0.0, 0.5])  # Chosen movie
        x_b = np.array([0.0, 1.0, 0.2])  # Unchosen movie
        
        updated_w = views.bradley_terry_update(w, x_a, x_b, lr=0.1)
        
        # Expect weight corresponding to x_a features to increase
        self.assertTrue(updated_w[0] > updated_w[1])
        self.assertIsInstance(updated_w, np.ndarray)

    def test_plackett_luce_probability(self):
        """Verify Plackett-Luce outputs a valid probability [0, 1]."""
        ranked_ids = [1, 2, 3]
        utility_scores = {1: 0.5, 2: 0.2, 3: 0.1}
        
        prob = views.plackett_luce_probability(ranked_ids, utility_scores)
        
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_get_or_init_user_vector(self):
        """Verify user weight vector initializes in session if missing."""
        request = self.factory.get('/')
        request.session = {}  # Simulate empty session
        
        w = views.get_or_init_user_vector(request)
        
        self.assertIn('user_w', request.session)
        self.assertEqual(len(w), self.features_dim)
        self.assertTrue((w == np.zeros(self.features_dim)).all())

    # --- View Tests ---

    def test_design1_get_request(self):
        """Test GET request to Design 1 (Binary Selection page)."""
        request = self.factory.get('/design1/')
        request.session = {}
        
        response = views.design1_view(request)
        
        self.assertEqual(response.status_code, 200)

    def test_design1_post_request(self):
        """Test POST request to Design 1 updates session vector via Bradley-Terry."""
        post_data = json.dumps({'chosen_id': 1, 'other_id': 2})
        request = self.factory.post(
            '/design1/', 
            data=post_data, 
            content_type='application/json'
        )
        request.session = {'user_w': [0.0, 0.0, 0.0]}
        
        response = views.design1_view(request)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify JSON payload response
        json_data = json.loads(response.content)
        self.assertEqual(json_data['status'], 'success')
        self.assertIn('updated_w', json_data)
        
        # Verify session was updated
        self.assertNotEqual(request.session['user_w'], [0.0, 0.0, 0.0])

    def test_design2_post_request(self):
        """Test POST request to Design 2 updates session vector via Plackett-Luce."""
        post_data = json.dumps({'ranked_ids': [1, 2, 3]})
        request = self.factory.post(
            '/design2/', 
            data=post_data, 
            content_type='application/json'
        )
        request.session = {'user_w': [0.0, 0.0, 0.0]}
        
        response = views.design2_view(request)
        
        self.assertEqual(response.status_code, 200)
        
        json_data = json.loads(response.content)
        self.assertEqual(json_data['status'], 'success')
        self.assertIn('ranking_probability', json_data)
        self.assertIn('updated_w', json_data)

    def test_recommendations_view(self):
        """Test recommendations page calculates utility scores and returns sorted recommendations."""
        request = self.factory.get('/recommendations/')
        # Custom weights favoring feature 0 (Movie A & C)
        request.session = {'user_w': [2.0, 0.0, 0.0]}
        
        response = views.recommendations_view(request, top_n=2)
        
        self.assertEqual(response.status_code, 200)