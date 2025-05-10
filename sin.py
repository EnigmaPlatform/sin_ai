import os
from pathlib import Path
from brain.model import SinModel
from brain.memory import SinMemory
from brain.trainer import SinTrainer

class Sin:
    def __init__(self):
        self.data_dir = Path("data")
        self.models_dir = self.data_dir / "models"
        self.conversations_dir = self.data_dir / "conversations"
        
        # Создание директорий
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        
        # Инициализация компонентов
        self.model = self._load_model()
        self.memory = SinMemory()
        self.trainer = SinTrainer(self.model)
        
        # Загрузка состояния
        self.load()
    
    def _load_model(self):
        model_path = self.models_dir / "sin_model.pt"
        if model_path.exists():
            return SinModel.load(model_path)
        return SinModel()
    
    def chat(self, user_input):
        """Обработка пользовательского ввода"""
        self.memory.add_interaction(user_input, "")
        
        # Формирование контекста
        context = self.memory.get_context()
        prompt = f"{context}\nSin:"
        
        # Генерация ответа
        response = self.model.generate_response(prompt)
        
        # Обновление памяти
        self.memory.add_interaction(user_input, response)
        return response
    
    def train(self):
        """Обучение на доступных данных"""
        loss = self.trainer.train(self.conversations_dir)
        self.save()
        return loss
    
    def save(self):
        """Сохранение состояния"""
        self.model.save(self.models_dir / "sin_model.pt")
        self.memory.save(self.data_dir / "memory.json")
    
    def load(self):
        """Загрузка состояния"""
        memory_path = self.data_dir / "memory.json"
        if memory_path.exists():
            self.memory.load(memory_path)
