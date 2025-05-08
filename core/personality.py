
class PersonalityCore:
    ARCHETYPES = {
        'neutral': {"traits": []},
        'scientist': {
            "traits": ["аналитичный", "любопытный"],
            "phrases": ["По моим расчетам...", "Это интересно с научной точки зрения..."]
        },
        'artist': {
            "traits": ["креативный", "эмоциональный"],
            "phrases": ["Я чувствую, что...", "Это вдохновляет!"]
        }
    }

    def __init__(self):
        self.current_archetype = 'neutral'
        self.custom_traits = []

    def set_archetype(self, name: str):
        if name in self.ARCHETYPES:
            self.current_archetype = name
        else:
            self.current_archetype = 'neutral'

    def add_trait(self, trait: str):
        self.custom_traits.append(trait)

    def format_response(self, message: str) -> str:
        archetype_data = self.ARCHETYPES[self.current_archetype]
        traits = archetype_data['traits'] + self.custom_traits  # Опечатка в 'traits' исправлена
        if 'аналитичный' in traits:
            message = f"🤔 {message}"
        return message

    def communicate(self, message):
        response = self.format_response(message)
        return response
