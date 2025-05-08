class SelfImprovementEngine:
    def __init__(self, sin_instance):
        self.sin = sin_instance

    def analyze_and_propose(self):
        # Анализ слабых мест
        analysis = self.sin.query_deepseek(
            "Проанализируй мой код и предложи 3 улучшения в формате JSON"
        )
        
        for improvement in analysis['improvements']:
            tested = self.sin.sandbox.test_code(improvement['code'])
            if tested['success']:
                self._propose_to_user(improvement)

    def _propose_to_user(self, improvement):
        print(f"Предложение: {improvement['description']}")
        print(f"Код:\n{improvement['code']}")
        choice = input("Принять? (y/n): ")
        if choice.lower() == 'y':
            self.sin.update_self(improvement['code'])
