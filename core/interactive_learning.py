class InteractiveTrainer:
    def __init__(self, model, data_dir="data/training"):
        self.model = model
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def start_training_session(self):
        """Интерактивный цикл обучения"""
        print("Режим обучения. Введите 'стоп' для выхода")
        
        while True:
            topic = input("\nТема для улучшения: ").strip()
            if topic.lower() == 'стоп':
                break
            
            # Диагностика слабых мест
            weaknesses = self._diagnose_weaknesses(topic)
            print(f"Найдены слабые места: {', '.join(weaknesses)}")
            
            # Генерация примеров
            for weakness in weaknesses:
                examples = self._generate_examples(topic, weakness)
                self._run_training_step(examples)
    
    def _diagnose_weaknesses(self, topic: str) -> list:
        """Выявление проблемных областей"""
        test_cases = [
            ("Определение", f"Дайте определение: {topic}"),
            ("Пример", f"Приведите пример использования {topic}"),
            ("Код", f"Напишите код демонстрирующий {topic}")
        ]
        
        weaknesses = []
        for case_type, prompt in test_cases:
            response = self.model.generate(prompt)
            score = self._evaluate_response(response, case_type)
            if score < 0.7:
                weaknesses.append(f"{case_type} (score: {score:.2f})")
        
        return weaknesses
    
    def _generate_examples(self, topic: str, weakness: str) -> list:
        """Генерация тренировочных данных"""
        if "Определение" in weakness:
            return self._generate_definitions(topic)
        elif "Пример" in weakness:
            return self._generate_usage_examples(topic)
        else:
            return self._generate_code_samples(topic)
