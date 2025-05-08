import unittest
from unittest.mock import patch, MagicMock
from core.network import SinNetwork
from pathlib import Path
import torch

class TestSinNetwork(unittest.TestCase):
    def setUp(self):
        self.network = SinNetwork()
    
    def test_initialization(self):
        self.assertIsNotNone(self.network.model)
        self.assertIsNotNone(self.network.tokenizer)
        self.assertEqual(self.network.device.type, 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_communicate(self):
        response = self.network.communicate("Hello")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_communicate_error(self):
        with patch.object(self.network.model, 'generate', side_effect=Exception("Test error")):
            with self.assertRaises(Exception):
                self.network.communicate("test")
    
    @patch('sin_ai.core.network.SinNetwork._extract_text_from_pdf')
    def test_learn_from_pdf(self, mock_extract):
        mock_extract.return_value = "Test text"
        test_pdf = Path("data/documents/test.pdf")
        test_pdf.touch()
        
        try:
            self.network.learn_from_file(str(test_pdf))
            mock_extract.assert_called_once()
        finally:
            test_pdf.unlink()
    
    def test_save_load_model(self):
        model_name = "test_model"
        self.network.save_model(model_name)
        
        try:
            loaded_model = self.network.model_manager.load_model(model_name, SinNetwork)
            self.assertIsNotNone(loaded_model)
        finally:
            self.network.model_manager.delete_model(model_name)

if __name__ == "__main__":
    unittest.main()
