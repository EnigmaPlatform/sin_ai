# Управление версиями
import os
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from transformers import AutoTokenizer

class ModelManager:
    def __init__(self, models_dir="data/models", training_dir="data/training"):
        self.models_dir = models_dir
        self.training_dir = training_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)
        
        # Инициализация подсистем
        self.tokenizer = AutoTokenizer.from_pretrained("DeepSeek/ai-base")
        self.experience = defaultdict(int)  # Уровни навыков
        self.feedback_log = []

    def save_model(self, model, metadata=None):
        """Расширенное сохранение модели с трекингом навыков"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"model_{version}.pt")
        
        # Сохраняем параметры модели + embeddings
        torch.save({
            'state_dict': model.state_dict(),
            'embeddings': model.get_embeddings(),
            'experience': dict(self.experience)
        }, model_path)
        
        # Метаданные с оценкой качества
        meta = {
            'version': version,
            'skills': self._assess_skills(model),
            'training_data': self._get_training_stats(),
            'feedback_score': self._calculate_feedback_score(),
            **metadata
        }
        
        with open(f"{model_path}.meta", 'w') as f:
            json.dump(meta, f)
        
        self._cleanup_old_models(max_keep=5)
        return version

    def load_model(self, version):
        """Загрузка с восстановлением контекста"""
        model_path = os.path.join(self.models_dir, f"model_{version}.pt")
        data = torch.load(model_path)
        
        model = SinNetwork()
        model.load_state_dict(data['state_dict'])
        model.load_embeddings(data['embeddings'])
        
        self.experience.update(data['experience'])
        return model

    def process_training_file(self, file_path, file_type=None):
        """Автоматическое определение типа данных и обработка"""
        if not file_type:
            file_type = self._detect_file_type(file_path)
        
        if file_type == "qa":
            return self._process_qa_file(file_path)
        elif file_type == "code":
            return self._process_code_file(file_path)
        elif file_type == "text":
            return self._process_text_file(file_path)

    def _process_qa_file(self, path):
        """Извлечение Q&A пар с аугментацией"""
        with open(path) as f:
            data = f.read()
        
        # Базовое извлечение
        pairs = re.findall(r"Q:(.+?)A:(.+?)(?=Q:|$)", data, re.DOTALL)
        
        # Аугментация - перефразирование вопросов
        augmented = []
        for q, a in pairs:
            augmented.append((q.strip(), a.strip()))
            paraphrased = self._paraphrase(q.strip())
            if paraphrased:
                augmented.append((paraphrased, a.strip()))
        
        return augmented

    def _process_code_file(self, path):
        """Анализ Python кода"""
        with open(path) as f:
            code = f.read()
        
        return {
            'original': code,
            'ast': self._parse_ast(code),
            'metrics': self._calculate_code_metrics(code)
        }

    def record_feedback(self, prompt, response, rating):
        """Запись обратной связи для RLHF"""
        self.feedback_log.append({
            'prompt': prompt,
            'response': response,
            'rating': rating,  # 1-5
            'timestamp': datetime.now().isoformat()
        })
        
        # Обновление весов на основе feedback
        self._adjust_weights(prompt, response, rating)

    def visualize_progress(self):
        """Генерация отчетов об обучении"""
        return {
            'skills': self._get_skill_levels(),
            'training_data_stats': self._get_training_stats(),
            'feedback_analysis': self._analyze_feedback()
        }

    # Вспомогательные методы
    def _detect_file_type(self, path):
        if path.endswith('.py'):
            return "code"
        elif re.search(r"(Q:|Question:)", open(path).read()[:1000]):
            return "qa"
        else:
            return "text"

    def _paraphrase(self, text):
        """Генерация вариаций вопроса (упрощенная версия)"""
        # Реальная реализация будет использовать LLM
        variations = [
            f"Перефразируй: {text}",
            f"Как еще можно спросить: {text}",
            f"Альтернативная формулировка: {text}"
        ]
        return np.random.choice(variations)

    def _adjust_weights(self, prompt, response, rating):
        """Алгоритм корректировки весов на основе оценок"""
        # Простейшая реализация - на практике нужен градиентный спуск
        adjustment = 0.1 * (rating - 3)  # Центрируем вокруг 3
        self.experience['dialogue_quality'] += adjustment
        self.experience['knowledge_depth'] += 0.05 * adjustment

    def _assess_skills(self, model):
        """Оценка навыков модели"""
        test_cases = {
            'code_generation': "Напиши функцию факториала на Python",
            'rewrite': "Перефразируй: 'Как дела?'",
            'context': "Продолжи диалог: 'Привет! Я изучаю AI...'"
        }
        
        return {skill: model.evaluate(task) for skill, task in test_cases.items()}

    def _get_training_stats(self):
        """Анализ использованных данных обучения"""
        counts = defaultdict(int)
        for file in os.listdir(self.training_dir):
            counts[self._detect_file_type(file)] += 1
        return dict(counts)
