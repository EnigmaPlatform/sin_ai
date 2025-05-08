from .base import SinPlugin

class ExamplePlugin(SinPlugin):
    def initialize(self, network):
        self.network = network
    
    def get_commands(self):
        return {
            'example': "Пример команды плагина",
            'calc': "Простой калькулятор"
        }
    
    def execute_command(self, command, args):
        if command == 'example':
            return "Это пример работы плагина!"
        elif command == 'calc':
            try:
                return f"Результат: {eval(args)}"  # Осторожно с eval!
            except:
                return "Ошибка вычисления"
