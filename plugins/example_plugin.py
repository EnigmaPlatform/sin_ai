from sin_ai.plugins.base import SinPlugin

class ExamplePlugin(SinPlugin):
    """Пример плагина для демонстрации возможностей"""
    
    def initialize(self, network):
        self.network = network
        self.counter = 0
    
    def get_commands(self):
        return {
            'example': "Пример команды плагина",
            'count': "Счетчик вызовов",
            'ask': "Задать вопрос через Sin"
        }
    
    def execute_command(self, command, args):
        if command == 'example':
            return "Это работает! Вы вызвали пример плагина."
        elif command == 'count':
            self.counter += 1
            return f"Счетчик: {self.counter}"
        elif command == 'ask':
            return self.network.communicate(args)
        return None
