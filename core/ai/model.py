import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from utils.logger import SinLogger
from utils.config import ConfigManager
from core.ai.memory import ContextMemory
from core.processing.text import TextProcessor
from core.processing.code import CodeAnalyzer
from core.ai.modifiers import CodeModifier

class SinModel(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.logger = SinLogger("SinModel")
        self.config = ConfigManager()
        
        # Инициализация компонентов
        self._initialize_models()
        self.memory = ContextMemory(
            short_term_size=self.config.get("memory.short_term_size", 10),
            long_term_size=self.config.get("memory.long_term_size", 1000)
        )
        self.text_processor = TextProcessor()
        self.code_analyzer = CodeAnalyzer()
        self.modifier = CodeModifier(self)
        
        # Состояние системы
        self.experience = 0
        self.skills = {
            "dialogue": 1.0,
            "coding": 1.0,
            "learning": 1.0
        }
    
    def _initialize_models(self):
        """Инициализация языковой модели и токенизатора"""
        model_name = self.config.get("model.name", "DeepSeek/ai-base")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.logger.info("Using CUDA acceleration")
    
    def generate(self, prompt: str, context: Optional[List[str]] = None, **kwargs) -> str:
        """Генерация ответа на запрос"""
        full_prompt = self._prepare_prompt(prompt, context)
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        output = self.model.generate(
            **inputs,
            max_length=self.config.get("model.max_length", 512),
            temperature=self.config.get("model.temperature", 0.7),
            **kwargs
        )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.memory.update(prompt, response)
        self._gain_experience(0.1)
        return response
    
    def learn(self, data: str, data_type: str = "text") -> Dict:
        """Обучение на предоставленных данных"""
        if data_type == "text":
            return self._learn_text(data)
        elif data_type == "code":
            return self._learn_code(data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def propose_change(self, new_code: str, description: str = "") -> Dict:
        """Предложить изменение собственного кода"""
        return self.modifier.propose_change(new_code, description)
    
    def apply_change(self, change_id: str) -> bool:
        """Применить предложенное изменение"""
        return self.modifier.apply_change(change_id)
    
    def save(self, path: str) -> None:
        """Сохранение состояния модели"""
        torch.save({
            "model_state": self.model.state_dict(),
            "memory": self.memory.state(),
            "experience": self.experience,
            "skills": self.skills
        }, path)
    
    def load(self, path: str) -> None:
        """Загрузка состояния модели"""
        data = torch.load(path)
        self.model.load_state_dict(data["model_state"])
        self.memory.load_state(data.get("memory"))
        self.experience = data.get("experience", 0)
        self.skills = data.get("skills", {})
    
    def _prepare_prompt(self, prompt: str, context: Optional[List[str]]) -> str:
        """Подготовка полного промпта с контекстом"""
        context = context or []
        memory_context = self.memory.retrieve(prompt)
        return "\n".join([*context, *memory_context, prompt])
    
    def _learn_text(self, text: str) -> Dict:
        """Обучение на текстовых данных"""
        analysis = self.text_processor.analyze(text)
        # Логика обучения на тексте...
        self._gain_experience(1.0)
        self.skills["dialogue"] = min(10.0, self.skills["dialogue"] + 0.5)
        return {"status": "success", "type": "text", "analysis": analysis}
    
    def _learn_code(self, code: str) -> Dict:
        """Обучение на коде"""
        analysis = self.code_analyzer.analyze(code)
        # Логика обучения на коде...
        self._gain_experience(2.0)
        self.skills["coding"] = min(10.0, self.skills["coding"] + 1.0)
        return {"status": "success", "type": "code", "analysis": analysis}
    
    def _gain_experience(self, amount: float):
        """Обновление уровня опыта"""
        self.experience += amount * self.skills["learning"]
        if self.experience >= 100 * (1.5 ** (self.skills["learning"] - 1)):
            self.skills["learning"] += 1
            self.logger.info(f"Level up! New learning level: {self.skills['learning']}")
