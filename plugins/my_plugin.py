from sin_ai.plugins.base import SinPlugin

class MyPlugin(SinPlugin):
    def initialize(self, network):
        self.network = network
    
    def get_commands(self):
        return {'greet': "Поздороваться"}
    
    def execute_command(self, command, args):
        if command == 'greet':
            return f"Привет, {args}!" if args else "Привет!"
