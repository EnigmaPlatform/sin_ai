import unittest
from unittest.mock import patch, MagicMock
from core.network.model import SinNetwork
from pathlib import Path
import torch

class TestSinNetwork(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model_name": "gpt2",  # Используем маленькую модель для тестов
            "max_length": 50,
            "memory_size": 100
        }
        self.model = SinNetwork(self.config)
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Создаем тестовые файлы
        (self.test_data_dir / "test.txt").write_text("Тестовый текст для обучения")
        (self.test_data_dir / "test.py").write_text("def hello():\n    return 'world'")

    def test_initialization(self):
        self.assertIsNotNone(self.model.tokenizer)
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.config['model_name'], "gpt2")

    def test_generate_response(self):
        with patch.object(self.model.model, 'generate') as mock_generate:
            mock_generate.return_value = torch.tensor([[1, 2, 3]])
            response = self.model.generate_response("Тестовый запрос")
            self.assertIsInstance(response, str)

    def test_learn_from_text_file(self):
        result = self.model.learn_from_file(str(self.test_data_dir / "test.txt"))
        self.assertEqual(result['status'], "success")
        self.assertEqual(result['type'], "text")

    def test_learn_from_code_file(self):
        result = self.model.learn_from_file(str(self.test_data_dir / "test.py"))
        self.assertEqual(result['status'], "success")
        self.assertEqual(result['type'], "code")

    def test_save_load(self):
        test_path = str(self.test_data_dir / "test_model.pt")
        self.model.save(test_path)
        
        new_model = SinNetwork(self.config)
        new_model.load(test_path)
        
        self.assertEqual(self.model.config, new_model.config)
        self.assertEqual(self.model.experience, new_model.experience)

    def test_propose_self_update(self):
        test_code = "def new_function():\n    return 'new'"
        result = self.model.propose_self_update(test_code)
        self.assertIn('status', result)
        self.assertIn('diff', result)

    def tearDown(self):
        # Удаляем тестовые файлы
        for file in self.test_data_dir.glob("*"):
            file.unlink()
        self.test_data_dir.rmdir()

if __name__ == '__main__':
    unittest.main()
