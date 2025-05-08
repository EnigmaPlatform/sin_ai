import math
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class LevelSystem:
    def __init__(self, level_file: str = "data/levels.json") -> None:
        self.level_file = Path(level_file)
        self.current_experience = 0
        self.current_level = 1
        self.total_learning_sessions = 0
        self.last_activity = datetime.now().isoformat()
        
        self._load_level_data()
    
    def add_experience(self, amount: int) -> None:
        """Добавление опыта пользователю.
        
        Args:
            amount: Количество добавляемого опыта (должно быть положительным)
        """
        if amount <= 0:
            raise ValueError("Amount of experience must be positive")
            
        self.current_experience += amount
        self._check_level_up()
        self.total_learning_sessions += 1
        self.last_activity = datetime.now().isoformat()
        self._save_level_data()
    
    def _check_level_up(self) -> None:
        """Проверяет, достигнут ли опыт для повышения уровня, и повышает уровень при необходимости."""
        while True:
            required_exp = self._exp_for_level(self.current_level + 1)
            if self.current_experience < required_exp:
                break
            self.current_level += 1
    
    @staticmethod
    def _exp_for_level(level: int) -> int:
        """Вычисляет количество опыта, необходимое для достижения указанного уровня.
        
        Args:
            level: Уровень, для которого рассчитывается опыт
            
        Returns:
            Количество опыта, необходимое для достижения уровня
        """
        if level < 1:
            raise ValueError("Level must be at least 1")
        return int(100 * (1.5 ** (level - 1)))
    
    def get_level_info(self) -> Dict[str, int | float | str]:
        """Возвращает текущую информацию об уровне пользователя.
        
        Returns:
            Словарь с информацией об уровне, опыте, прогрессе и активности
        """
        return {
            'level': self.current_level,
            'experience': self.current_experience,
            'next_level_exp': self._exp_for_level(self.current_level + 1),
            'progress': self._level_progress(),
            'total_sessions': self.total_learning_sessions,
            'last_activity': self.last_activity
        }
    
    def _level_progress(self) -> float:
        """Рассчитывает прогресс до следующего уровня в диапазоне от 0 до 1.
        
        Returns:
            Прогресс до следующего уровня (0.0 - 1.0)
        """
        current_level_exp = self._exp_for_level(self.current_level)
        next_level_exp = self._exp_for_level(self.current_level + 1)
        
        if next_level_exp == current_level_exp:
            return 0.0
            
        return min(1.0, max(0.0, (self.current_experience - current_level_exp) / 
                   (next_level_exp - current_level_exp))
    
    def _load_level_data(self) -> None:
        """Загружает данные об уровне из файла. Если файл не существует или поврежден, 
        используются значения по умолчанию."""
        try:
            if self.level_file.exists():
                with open(self.level_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_experience = data.get('current_experience', 0)
                    self.current_level = data.get('current_level', 1)
                    self.total_learning_sessions = data.get('total_learning_sessions', 0)
                    self.last_activity = data.get('last_activity', datetime.now().isoformat())
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading level data: {e}. Using default values.")
    
    def _save_level_data(self) -> None:
        """Сохраняет текущие данные об уровне в файл."""
        data = {
            'current_experience': self.current_experience,
            'current_level': self.current_level,
            'total_learning_sessions': self.total_learning_sessions,
            'last_activity': self.last_activity
        }
        
        try:
            self.level_file.parent.mkdir(exist_ok=True, parents=True)
            with open(self.level_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving level data: {e}")
