class DeepSeekTrainer:
    def __init__(self, sin_instance):
        self.sin = sin_instance
        self.task_types = {
            'programming': self._train_programming,
            'communication': self._train_communication
        }

    def generate_task(self):
    prompt = f"""
    Сгенерируй задачу для ИИ с характеристиками: 
    {self.sin.personality.current_mode}
    """
    return self.sin.query_deepseek(prompt)

    async def train(self, task_type: str):
        # DeepSeek генерирует задачу
        task = self.sin.query_deepseek(f"Сгенерируй {task_type} задачу для обучения ИИ")
        
        # Sin пытается выполнить
        response = self.sin.communicate(task)
        
        # DeepSeek оценивает ответ
        evaluation = self.sin.query_deepseek(
            f"Оцени ответ '{response}' на задачу '{task}' по шкале 0-100"
        )
        
        # Обработка оценки
        score = self._parse_score(evaluation)
        if score >= 70:
            self.sin.level_system.add_experience(score)
            self.sin.memory.add_memory(f"Успешное решение: {task}", importance=0.9)
        else:
            correction = self.sin.query_deepseek(f"Покажи правильное решение для: {task}")
            self.sin.learn_from_text(correction)
