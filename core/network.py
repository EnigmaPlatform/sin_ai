from __future__ import annotations
from typing import List, Dict, Optional, Union, Any
import torch
import torch.nn as nn
import logging
from pathlib import Path
import json
from datetime import datetime
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from PyPDF2 import PdfReader
from docx import Document
import ast

# Относительные импорты
from models.model_manager import ModelManager
from core.emotions import EmotionEngine
from core.memory import MemorySystem
from core.api_handler import DeepSeekAPIHandler
from core.code_analyzer import CodeAnalyzer
from core.sandbox import CodeSandbox
from core.level_system import LevelSystem
from core.personality import PersonalityCore
from core.deepseek_trainer import DeepSeekTrainer
from core.utils import validate_input, validate_text, validate_file_path, validate_language

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SinNetwork(nn.Module):
    """
    Основной класс нейросетевой архитектуры Sin AI.
    """
    def __init__(self, model_name: str = "sin_base"):
        super(SinNetwork, self).__init__()
        self.model_name = model_name
        self.emotions = EmotionEngine()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализация компонентов
        self.config = GPT2Config.from_pretrained("gpt2-medium")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = GPT2Model(self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        # Подсистемы (ленивая загрузка визуализатора)
        self.memory = MemorySystem()
        self.learning_engine = LearningEngine(self)
        self.api_handler = DeepSeekAPIHandler()
        self.code_analyzer = CodeAnalyzer()
        self.sandbox = CodeSandbox()
        self.level_system = LevelSystem()
        self._visualizer = None  # Ленивая загрузка
        self.model_manager = ModelManager()
        self.deepseek_trainer = DeepSeekTrainer(self)
        
        # Состояние
        self.current_context = []
        self.learning_progress = 0
        self.is_learning = False
    
    @property
    def visualizer(self):
        if self._visualizer is None:
            from ui.visualizer import TrainingVisualizer
            self._visualizer = TrainingVisualizer()
        return self._visualizer
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def communicate(self, message: str) -> str:
        """
        Основной метод для общения с пользователем.
        
        Пример:
        >>> response = sin.communicate("Какая сегодня погода?")
        >>> print(response)
        """
        # Обновление контекста
        try:
            self.current_context.append(f"User: {message}")
            
            # Подготовка ввода
            input_text = "\n".join(self.current_context[-5:])
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

        except Exception as e:
            logger.error(f"Communication error: {str(e)}")
            return "Sorry, an error occurred while processing your message"
    
    @validate_input(text=validate_text)
    def learn_from_text(self, text: str) -> None:
        """
        Обучение на текстовых данных.
        
        Пример:
        >>> sin.learn_from_text("ИИ - это система, способная выполнять задачи...")
        """
        self.is_learning = True
        try:
            self.learning_engine.train_on_text(text)
            self.level_system.add_experience(10)
            logger.info("Successfully learned from text")
        except Exception as e:
            logger.error(f"Error learning from text: {e}")
        finally:
            self.is_learning = False
    
    @validate_input(file_path=validate_file_path)
    def learn_from_file(self, file_path: str) -> None:
        """
        Обучение из файла (поддерживает txt, docx, pdf).
        
        Пример:
        >>> sin.learn_from_file("data/document.pdf")
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        self.is_learning = True
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.learn_from_text(text)
            elif file_path.endswith(('.doc', '.docx')):
                text = self._extract_text_from_word(file_path)
                self.learn_from_text(text)
            elif file_path.endswith('.pdf'):
                text = self._extract_text_from_pdf(file_path)
                self.learn_from_text(text)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
            
            self.level_system.add_experience(20)
        except Exception as e:
            logger.error(f"Error learning from file: {e}")
        finally:
            self.is_learning = False

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Извлечение текста из PDF"""
        from PyPDF2 import PdfReader
        text = []
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)

    def learn_from_json(self, 
                      file_path: str, 
                      text_fields: List[str] = None,
                      context_field: str = None) -> None:
        """
        Обучение на структурированном JSON
        
        :param file_path: Путь к JSON-файлу
        :param text_fields: Поля для извлечения текста (например, ["text", "answer"])
        :param context_field: Поле контекста (например, "situation")
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        data = self._load_json(path)
        processed_text = self._process_json_data(data, text_fields, context_field)
        self.learn_from_text(processed_text)

    def _load_json(self, path: Path) -> Union[Dict, List]:
        """Безопасная загрузка JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")

    def _process_json_data(self, 
                         data: Union[Dict, List], 
                         text_fields: List[str],
                         context_field: str) -> str:
        """Рекурсивная обработка JSON-структур"""
        result = []
        
        if isinstance(data, list):
            for item in data:
                result.append(self._process_json_data(item, text_fields, context_field))
        elif isinstance(data, dict):
            # Извлечение контекста
            context = data.get(context_field, "") if context_field else ""
            
            # Извлечение текстовых полей
            texts = []
            for field in text_fields or []:
                if field in data:
                    texts.append(str(data[field]))
            
            if texts:
                result.append(f"{context}\n" + "\n".join(texts) if context else "\n".join(texts))
        
        return "\n\n".join(filter(None, result))
    
    @validate_input(language=validate_language)
    def learn_from_code(self, code: str, language: str = 'python') -> None:
        """
        Обучение на примере кода.
        
        Пример:
        >>> code = \"\"\"
        def factorial(n):
            return 1 if n == 0 else n * factorial(n-1)
        \"\"\"
        >>> sin.learn_from_code(code, "python")
        """
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
        """
        Запрос к DeepSeek API.
        
        Пример:
        >>> response = sin.query_deepseek("Объясни теорию относительности")
        >>> print(response)
        """
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
        """
        Сохранение текущей модели.
        
        Пример:
        >>> sin.save_model("my_model_v1")
        """
        model_name = model_name or self.model_name
        self.model_manager.save_model(self, model_name)
        logger.info(f"Model saved as {model_name}")
    
    def delete_model(self, model_name: str) -> None:
        """
        Удаление сохраненной модели.
        
        Пример:
        >>> sin.delete_model("old_model")
        """
        self.model_manager.delete_model(model_name)
        logger.info(f"Model {model_name} deleted")
    
    def propose_feature(self) -> Dict:
        """
        Предложение нового функционала.
        
        Пример:
        >>> feature = sin.propose_feature()
        >>> print(feature['description'])
        """
        feature = {
            'description': '',
            'code': '',
            'tests': '',
            'benefits': ''
        }
        
        current_capabilities = self._analyze_capabilities()
        feature_suggestion = self._generate_feature_suggestion(current_capabilities)
        
        generated = self.query_deepseek(f"Generate Python code for: {feature_suggestion}")
        
        feature['description'] = feature_suggestion
        feature['code'] = generated.get('code', '')
        feature['tests'] = generated.get('tests', '')
        feature['benefits'] = generated.get('benefits', '')
        
        return feature
    
    def test_feature(self, feature_code: str, test_code: str) -> Dict:
        """
        Тестирование нового функционала в песочнице.
        
        Пример:
        >>> test_result = sin.test_feature("def add(a,b): return a+b", 
        ...                                "assert add(2,3) == 5")
        >>> print(test_result['test_passed'])
        """
        return self.sandbox.test_feature(feature_code, test_code)
    
    def update_self(self, new_code: str) -> bool:
        """
        Обновление собственного кода.
        
        Пример:
        >>> new_code = \"\"\"
        def new_method(self):
            return "Это новая функциональность!"
        \"\"\"
        >>> if sin.update_self(new_code):
        ...     print("Обновление успешно!")
        """
        try:
            test_result = self.sandbox.test_code(new_code)
            if not test_result.get('success', False):
                return False
            
            self._apply_code_update(new_code)
            self.level_system.add_experience(50)
            return True
        except Exception as e:
            logger.error(f"Error updating code: {e}")
            return False
    
    def get_learning_progress(self) -> Dict:
        """
        Получение информации о ходе обучения.
        
        Пример:
        >>> progress = sin.get_learning_progress()
        >>> print(f"Уровень: {progress['level']}")
        """
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
        """Применение обновления кода с проверками и откатами"""
        try:
            # 1. Создаем резервную копию текущего состояния
            backup = {
                'model_state': self.model.state_dict(),
                'tokenizer': self.tokenizer.get_vocab(),
                'config': self.config.to_dict()
            }
            
            # 2. Пытаемся применить изменения
            temp_globals = {}
            temp_locals = {}
            
            exec(new_code, temp_globals, temp_locals)
            
            # 3. Проверяем обязательные компоненты
            required_components = ['model', 'tokenizer', 'config']
            for component in required_components:
                if component not in temp_locals:
                    raise ValueError(f"Новый код не содержит обязательного компонента: {component}")
            
            # 4. Валидация новых компонентов
            if not isinstance(temp_locals['model'], type(self.model)):
                raise TypeError("Новая модель имеет несовместимый тип")
            
            # 5. Применяем изменения
            self.model = temp_locals['model'].to(self.device)
            self.tokenizer = temp_locals['tokenizer']
            self.config = temp_locals['config']
            
            # 6. Проверяем работоспособность
            test_input = "Test input"
            input_ids = self.tokenizer.encode(test_input, return_tensors="pt").to(self.device)
            outputs = self.model(input_ids)
            
            if outputs.last_hidden_state.shape[0] != 1:
                raise ValueError("Новая модель работает некорректно")
            
            logger.info("Код успешно обновлен")
            
        except Exception as e:
            # Откат изменений в случае ошибки
            if 'backup' in locals():
                self.model.load_state_dict(backup['model_state'])
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.config = GPT2Config.from_dict(backup['config'])
                logger.info("Выполнен откат к предыдущей версии")
            
            logger.error(f"Ошибка при обновлении кода: {str(e)}")
            raise
