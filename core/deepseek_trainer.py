class DeepSeekTrainer:
    def __init__(self, sin_instance):
        self.sin = sin_instance
        self.task_types = {
            'programming': self._train_programming,
            'communication': self._train_communication
        }

    def _train_programming(self):
        """Обучение программированию"""
        task = self.sin.query_deepseek("Сгенерируй задачу по программированию")
        response = self.sin.communicate(task)
        
        evaluation = self.sin.query_deepseek(
            f"Оцени ответ '{response}' на задачу '{task}' по шкале 0-100"
        )
        return self._process_evaluation(evaluation)

    def _train_communication(self):
        """Обучение коммуникации"""
        task = self.sin.query_deepseek("Сгенерируй диалоговую задачу")
        response = self.sin.communicate(task)
        
        evaluation = self.sin.query_deepseek(
            f"Оцени ответ '{response}' на диалоговую задачу '{task}'"
        )
        return self._process_evaluation(evaluation)

    def _process_evaluation(self, evaluation):
        """Обработка оценки от DeepSeek"""
        try:
            score = int(evaluation)
            if score >= 70:
                self.sin.level_system.add_experience(score)
                return True
            return False
        except ValueError:
            return False

    def train(self, task_type: str):
        """Основной метод обучения"""
        if task_type not in self.task_types:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return self.task_types[task_type]()
