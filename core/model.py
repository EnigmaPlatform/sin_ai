class SinNetwork:
    def __init__(self):
        self.context_manager = ContextManager()
        self.self_modifier = AdvancedSelfModifier()
        self.hybrid_learner = HybridLearner(self)
        self.interactive_trainer = InteractiveTrainer(self)
        
        # Инициализация моделей
        self.dialogue_model = load_pretrained("dialogue")
        self.code_model = load_pretrained("code")
        self.rewrite_model = load_pretrained("rewrite")
    
    def process_input(self, input_data):
        """Основной метод обработки ввода"""
        # Определение типа ввода
        processor = self._get_processor(input_data)
        
        # Обработка с учетом контекста
        context = self.context_manager.get_relevant_context(input_data)
        response = processor.generate(input_data, context)
        
        # Обновление состояния
        self.context_manager.update_context(input_data, response)
        return response
    
    def _get_processor(self, input_data):
        """Выбор специализированной модели"""
        if self._looks_like_code(input_data):
            return self.code_model
        elif self._requires_rewrite(input_data):
            return self.rewrite_model
        else:
            return self.dialogue_model
