# tests/test_api_handler.py

import unittest
from unittest.mock import patch
from sin_ai.core.api_handler import DeepSeekAPIHandler

class TestAPIHandler(unittest.TestCase):
    @patch('requests.Session.post')
    def test_query_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'choices': [{'message': {'content': 'test'}}]}
        
        handler = DeepSeekAPIHandler()
        result = handler.query("test")
        self.assertEqual(result['response'], 'test')
