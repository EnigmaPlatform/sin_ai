# Интерфейс командной строки
import cmd
from core.network.model import SinNetwork

class SinCLI(cmd.Cmd):
    prompt = "(Sin) "
    
    def __init__(self):
        super().__init__()
        self.model = SinNetwork()
    
    def do_chat(self, arg):
        """Начать диалог: chat [сообщение]"""
        response = self.model.generate_response(arg)
        print(f"Sin: {response}")
    
    def do_train(self, arg):
        """Обучить на файле: train [путь]"""
        self.model.learn_from_file(arg)
        print("Обучение завершено!")
    
    def do_update(self, arg):
        """Предложить изменение кода: update [файл_с_кодом]"""
        with open(arg) as f:
            code = f.read()
        result = self.model.propose_self_update(code)
        print("Предлагаемые изменения:")
        print(result['diff'])
        
        if input("Применить? (y/N): ").lower() == 'y':
            self.model.apply_update(code)
            print("Изменения применены!")

if __name__ == '__main__':
    SinCLI().cmdloop()
