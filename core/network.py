import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from ..models.model_manager import ModelManager
from .learning import LearningEngine
from .memory import MemorySystem
from .api_handler import DeepSeekAPIHandler
from .code_analyzer import CodeAnalyzer
from .sandbox import CodeSandbox
from .level_system import LevelSystem
from ..ui.visualizer import TrainingVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SinNetwork(nn.Module):
    def __init__(self, model_name: str = "sin_base"):
        super(SinNetwork, self).__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализация компонентов
        self.config = GPT2Config.from_pretrained("gpt2-medium")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = GPT2Model(self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        # Подсистемы
        self.memory = MemorySystem()
        self.learning_engine = LearningEngine(self)
        self.api_handler = DeepSeekAPIHandler()
        self.code_analyzer = CodeAnalyzer()
        self.sandbox = CodeSandbox()
        self.level_system = LevelSystem()
        self.visualizer = TrainingVisualizer()
        self.model_manager = ModelManager()
        
        # Состояние
        self.current_context = []
        self.learning_progress = 0
        self.is_learning = False
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def communicate(self, message: str) -> str:
        """Основной метод для общения с пользователем"""
        # Обновление контекста
        self.current_context.append(f"User: {message}")
        
        # Подготовка ввода
        input_text = "\n".join(self.current_context[-5:])  # Берем последние 5 сообщений для контекста
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Генерация ответа
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(input_text):].strip()
        
        # Обновление контекста
        self.current_context.append(f"Sin: {response}")
        
        return response
    
    def learn_from_text(self, text: str) -> None:
        """Обучение на текстовых данных"""
        self.is_learning = True
        try:
            self.learning_engine.train_on_text(text)
            self.level_system.add_experience(10)
            logger.info("Successfully learned from text")
        except Exception as e:
            logger.error(f"Error learning from text: {e}")
        finally:
            self.is_learning = False
    
    def learn_from_file(self, file_path: str) -> None:
        """Обучение из файла"""
        self.is_learning = True
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.learn_from_text(text)
            elif file_path.endswith(('.doc', '.docx')):
                text = self._extract_text_from_word(file_path)
                self.learn_from_text(text)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
            
            self.level_system.add_experience(20)
        except Exception as e:
            logger.error(f"Error learning from file: {e}")
        finally:
            self.is_learning = False
    
    def learn_from_code(self, code: str, language: str = 'python') -> None:
        """Обучение на примере кода"""
        self.is_learning = True
        try:
            analysis = self.code_analyzer.analyze(code, language)
            self.learning_engine.train_on_code(analysis)
            self.level_system.add_experience(30)
            logger.info(f"Successfully learned from {language} code")
        except Exception as e:
            logger.error(f"Error learning from code: {e}")
        finally:
            self.is_learning = False
    
    def query_deepseek(self, query: str) -> str:
        """Запрос к DeepSeek API"""
        self.is_learning = True
        try:
            response = self.api_handler.query(query)
            self.learning_engine.process_api_response(response)
            self.level_system.add_experience(15)
            return response
        except Exception as e:
            logger.error(f"Error querying DeepSeek: {e}")
            return f"Error: {str(e)}"
        finally:
            self.is_learning = False
    
    def save_model(self, model_name: Optional[str] = None) -> None:
        """Сохранение текущей модели"""
        model_name = model_name or self.model_name
        self.model_manager.save_model(self, model_name)
        logger.info(f"Model saved as {model_name}")
    
    def delete_model(self, model_name: str) -> None:
        """Удаление сохраненной модели"""
        self.model_manager.delete_model(model_name)
        logger.info(f"Model {model_name} deleted")
    
    def propose_feature(self) -> Dict:
        """Предложение нового функционала"""
        feature = {
            'description': '',
            'code': '',
            'tests': '',
            'benefits': ''
        }
        
        # Анализ текущих возможностей и предложение улучшений
        current_capabilities = self._analyze_capabilities()
        feature_suggestion = self._generate_feature_suggestion(current_capabilities)
        
        # Генерация кода и тестов
        generated = self.query_deepseek(f"Generate Python code for: {feature_suggestion}")
        
        feature['description'] = feature_suggestion
        feature['code'] = generated.get('code', '')
        feature['tests'] = generated.get('tests', '')
        feature['benefits'] = generated.get('benefits', '')
        
        return feature
    
    def test_feature(self, feature_code: str, test_code: str) -> Dict:
        """Тестирование нового функционала в песочнице"""
        return self.sandbox.test_feature(feature_code, test_code)
    
    def update_self(self, new_code: str) -> bool:
        """Обновление собственного кода"""
        try:
            # Тестируем новый код
            test_result = self.sandbox.test_code(new_code)
            if not test_result.get('success', False):
                return False
            
            # Если тесты пройдены, применяем изменения
            self._apply_code_update(new_code)
            self.level_system.add_experience(50)
            return True
        except Exception as e:
            logger.error(f"Error updating code: {e}")
            return False
    
    def get_learning_progress(self) -> Dict:
        """Получение информации о ходе обучения"""
        return {
            'progress': self.learning_progress,
            'level': self.level_system.current_level,
            'experience': self.level_system.current_experience,
            'is_learning': self.is_learning,
            'learning_speed': self.learning_engine.learning_speed,
            'recent_topics': self.memory.get_recent_topics(3)
        }
    
    def _extract_text_from_word(self, file_path: str) -> str:
        """Извлечение текста из Word документов"""
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _analyze_capabilities(self) -> List[str]:
        """Анализ текущих возможностей"""
        return [
            "Text understanding",
            "Contextual conversation",
            "Learning from text",
            "Learning from code",
            "API integration"
        ]
    
    def _generate_feature_suggestion(self, capabilities: List[str]) -> str:
        """Генерация предложения по улучшению"""
        prompt = f"Current capabilities: {', '.join(capabilities)}. Suggest one useful feature enhancement."
        response = self.query_deepseek(prompt)
        return response.get('suggestion', 'No suggestion generated')
    
    def _apply_code_update(self, new_code: str) -> None:
        """Применение обновления кода"""
        # Здесь должна быть логика безопасного обновления кода
        # В реальной реализации это было бы сложнее с проверками и откатами
        logger.warning("Code update functionality is simplified in this example")
