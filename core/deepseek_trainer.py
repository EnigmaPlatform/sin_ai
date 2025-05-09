import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DeepSeekTrainer:
    def __init__(self, sin_instance):
        self.sin = sin_instance
        self.task_types = {
            'programming': self._train_programming,
            'communication': self._train_communication
        }

    def _train_programming(self) -> bool:
        """Метод тренировки программированию"""
        try:
            task = self.sin.query_deepseek("Сгенерируй задачу по программированию на Python")
            if not task or isinstance(task, dict):
                logger.error("Invalid task received from API")
                return False
                
            response = self.sin.communicate(task)
            evaluation = self._process_evaluation(response)
            
            if evaluation >= 70:
                self.sin.level_system.add_experience(evaluation)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Programming training error: {str(e)}")
            return False

    def _train_communication(self) -> bool:
        """Метод тренировки коммуникации"""
        try:
            task = self.sin.query_deepseek("Сгенерируй диалоговую ситуацию для тренировки общения")
            if not task or isinstance(task, dict):
                logger.error("Invalid task received from API")
                return False
                
            response = self.sin.communicate(task)
            evaluation = self._process_evaluation(response)
            
            if evaluation >= 70:
                self.sin.level_system.add_experience(evaluation)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Communication training error: {str(e)}")
            return False

    def _process_evaluation(self, response: str) -> int:
        """Обработка оценки ответа"""
        try:
            if isinstance(response, dict):
                return 0
            return min(100, max(0, len(response) // 2))  # Простая метрика качества
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return 0

    def train(self, task_type: str) -> bool:
        """
        Основной метод тренировки
        
        Args:
            task_type: Тип тренировки ('programming' или 'communication')
            
        Returns:
            bool: True если тренировка успешна, False если есть ошибки
        """
        try:
            if task_type not in self.task_types:
                raise ValueError(f"Unknown task type: {task_type}. Use 'programming' or 'communication'")
                
            training_method = self.task_types[task_type]
            return training_method()
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False
