import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List, Dict
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
from ..ui.visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class LearningEngine:
    def __init__(self, network):
        self.network = network
        self.learning_rate = 5e-5
        self.batch_size = 4
        self.epochs = 3
        self.learning_speed = 1.0
        self.visualizer = TrainingVisualizer()
    
    def train_on_text(self, text: str) -> None:
        """Обучение на текстовых данных"""
        # Разделение текста на части
        chunks = self._chunk_text(text)
        dataset = TextDataset(chunks, self.network.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Настройка оптимизатора
        optimizer = AdamW(self.network.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * self.epochs
        )
        
        # Обучение
        self.network.train()
        total_steps = len(dataloader) * self.epochs
        progress_bar = tqdm(total=total_steps, desc="Training on text")
        
        for epoch in range(self.epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.network.device)
                attention_mask = batch['attention_mask'].to(self.network.device)
                
                outputs = self.network(input_ids, attention_mask=attention_mask)
                loss = self._calculate_loss(outputs, input_ids)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Обновление прогресса
                progress_bar.update(1)
                self.network.learning_progress = (progress_bar.n / progress_bar.total) * 100
                self.visualizer.update_progress(self.network.learning_progress)
        
        progress_bar.close()
        self.network.eval()
    
    def train_on_code(self, code_analysis: Dict) -> None:
    """Специализированное обучение на анализе кода"""
    try:
        language = code_analysis['language'].lower()
        code = code_analysis['code']
        analysis = code_analysis['analysis']
        
        # 1. Создаем специализированные примеры для обучения
        examples = []
        
        # Примеры для синтаксиса языка
        examples.append(f"Синтаксис {language}:\n{self._generate_syntax_examples(language)}")
        
        # Примеры из конкретного кода
        examples.append(f"Пример кода на {language}:\n{code}")
        
        # Примеры с объяснениями
        if analysis:
            examples.append(f"Анализ кода:\n{analysis}")
        
        # Добавляем примеры использования
        examples.append(f"Примеры использования конструкций из кода:\n{self._generate_usage_examples(code, language)}")
        
        # 2. Специальная обработка для разных языков
        if language == 'python':
            # Добавляем документацию Python
            examples.append("Python documentation:\n" + self._extract_python_docs(code))
        elif language == 'javascript':
            # Добавляем примеры JS специфичные
            examples.append("JavaScript специфика:\n" + self._extract_js_specifics(code))
        
        # 3. Создаем специализированные задачи для обучения
        training_text = "\n\n".join(examples)
        
        # 4. Настраиваем специальные параметры обучения для кода
        original_lr = self.learning_rate
        self.learning_rate = 3e-5  # Более высокая скорость обучения для кода
        
        # 5. Обучаем на подготовленных данных
        self.train_on_text(training_text)
        
        # Возвращаем исходную скорость обучения
        self.learning_rate = original_lr
        
        logger.info(f"Специализированное обучение на {language} коде завершено")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении на коде: {str(e)}")
        raise

def _generate_syntax_examples(self, language: str) -> str:
    """Генерация примеров синтаксиса для языка"""
    syntax_map = {
        'python': [
            "Условные операторы: if x > 0: ... elif x == 0: ... else: ...",
            "Циклы: for i in range(10): ... while condition: ...",
            "Функции: def func(arg): ...",
            "Классы: class MyClass: ..."
        ],
        'javascript': [
            "Функции: function myFunc() { ... } или const fn = () => { ... }",
            "Циклы: for(let i=0; i<10; i++) { ... } while(condition) { ... }",
            "Классы: class MyClass { ... }"
        ]
    }
    return "\n".join(syntax_map.get(language, []))

def _generate_usage_examples(self, code: str, language: str) -> str:
    """Генерация примеров использования конструкций из кода"""
    # Здесь можно добавить анализ кода и создание примеров
    return f"Использование конструкций из предоставленного кода на {language}"

def _extract_python_docs(self, code: str) -> str:
    """Извлечение документации Python из кода"""
    try:
        import ast
        
        docs = []
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docs.append(f"{node.name}:\n{docstring}")
        
        return "\n\n".join(docs) if docs else "No Python docstrings found"
    except:
        return "Could not extract Python documentation"
    
    def process_api_response(self, response: Dict) -> None:
    """Расширенная обработка ответа от API для обучения"""
    try:
        # 1. Обработка текстовых знаний
        if 'knowledge' in response:
            # Добавляем мета-информацию об источнике
            source = response.get('source', 'DeepSeek API')
            text = f"Знания от {source}:\n{response['knowledge']}"
            self.train_on_text(text)
        
        # 2. Обработка примеров кода
        if 'code' in response:
            language = response.get('language', 'python')
            explanation = response.get('explanation', '')
            
            # Создаем структурированный пример для обучения
            code_example = (
                f"Пример кода на {language}:\n```{language}\n{response['code']}\n```\n"
                f"Объяснение:\n{explanation}\n"
                f"Ключевые концепции:\n{self._extract_concepts(response['code'], language)}"
            )
            self.train_on_text(code_example)
        
        # 3. Обработка структурированных данных
        if 'structured_data' in response:
            self._process_structured_data(response['structured_data'])
        
        # 4. Обработка исправлений ошибок
        if 'error_fixes' in response:
            self._learn_from_error_fixes(response['error_fixes'])
            
    except Exception as e:
        logger.error(f"Ошибка при обработке API ответа: {str(e)}")

def _extract_concepts(self, code: str, language: str) -> str:
    """Извлечение ключевых концепций из кода"""
    concepts = {
        'python': {
            'def ': 'Функции',
            'class ': 'Классы',
            'import ': 'Импорты',
            'for ': 'Циклы for',
            'while ': 'Циклы while',
            'try:': 'Обработка исключений'
        },
        'javascript': {
            'function ': 'Функции',
            'class ': 'Классы',
            'import ': 'Импорты',
            'for (': 'Циклы for',
            'while (': 'Циклы while',
            'try {': 'Обработка исключений'
        }
    }
    
    found = set()
    for pattern, concept in concepts.get(language, {}).items():
        if pattern in code:
            found.add(concept)
    
    return ", ".join(found) if found else "Базовые конструкции"

def _process_structured_data(self, data: Dict) -> None:
    """Обработка структурированных данных"""
    if 'entities' in data:
        entities_text = "\n".join(f"{e['type']}: {e['name']}" for e in data['entities'])
        self.train_on_text(f"Сущности:\n{entities_text}")
    
    if 'relations' in data:
        relations_text = "\n".join(f"{r['source']} -> {r['target']}: {r['type']}" for r in data['relations'])
        self.train_on_text(f"Отношения:\n{relations_text}")

def _learn_from_error_fixes(self, fixes: List[Dict]) -> None:
    """Обучение на исправлениях ошибок"""
    for fix in fixes:
        example = (
            f"Ошибка: {fix['error']}\n"
            f"Неправильно: {fix['incorrect']}\n"
            f"Правильно: {fix['correct']}\n"
            f"Объяснение: {fix.get('explanation', '')}"
        )
        self.train_on_text(example)
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Разделение текста на части"""
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def _calculate_loss(self, outputs, input_ids):
        """Вычисление потерь для языковой модели"""
        # Упрощенная версия - в реальной реализации было бы сложнее
        logits = torch.matmul(outputs, self.network.model.wte.weight.t())
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
