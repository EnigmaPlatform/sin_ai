import math
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

class LevelSystem:
    def __init__(self, level_file: str = "data/levels.json"):
        self.level_file = Path(level_file)
        self.current_experience = 0
        self.current_level = 1
        self.total_learning_sessions = 0
        self.last_activity = datetime.now().isoformat()
        
        self._load_level_data()
    
    def add_experience(self, amount: int) -> None:
        """Добавление опыта"""
        self.current_experience += amount
        self._check_level_up()
        self.total_learning_sessions += 1
        self.last_activity = datetime.now().isoformat()
        self._save_level_data()
    
    def _check_level_up(self) -> None:
        """Проверка повышения уровня"""
        required_exp = self._exp_for_level(self.current_level + 1)
        while self.current_experience >= required_exp:
            self.current_level += 1
            required_exp = self._exp_for_level(self.current_level + 1)
    
    def _exp_for_level(self, level: int) -> int:
        """Опыт, необходимый для уровня"""
        return int(100 * (1.5 ** (level - 1)))
    
    def get_level_info(self) -> Dict:
        """Получение информации об уровне"""
        return {
            'level': self.current_level,
            'experience': self.current_experience,
            'next_level_exp': self._exp_for_level(self.current_level + 1),
            'progress': self._level_progress(),
            'total_sessions': self.total_learning_sessions,
            'last_activity': self.last_activity
        }
    
    def _level_progress(self) -> float:
        """Прогресс до следующего уровня (0-1)"""
        current_level_exp = self._exp_for_level(self.current_level)
        next_level_exp = self._exp_for_level(self.current_level + 1)
        return (self.current_experience - current_level_exp) / (next_level_exp - current_level_exp)
    
    def _load_level_data(self) -> None:
        """Загрузка данных об уровне"""
        try:
            if self.level_file.exists():
                with open(self.level_file, 'r') as f:
                    data = json.load(f)
                    self.current_experience = data.get('current_experience', 0)
                    self.current_level = data.get('current_level', 1)
                    self.total_learning_sessions = data.get('total_learning_sessions', 0)
                    self.last_activity = data.get('last_activity', datetime.now().isoformat())
        except Exception:
            pass
    
    def _save_level_data(self) -> None:
        """Сохранение данных об уровне"""
        data = {
            'current_experience': self.current_experience,
            'current_level': self.current_level,
            'total_learning_sessions': self.total_learning_sessions,
            'last_activity': self.last_activity
        }
        
        self.level_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.level_file, 'w') as f:
            json.dump(data, f, indent=2)
