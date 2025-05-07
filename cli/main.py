import cmd
import argparse
from pathlib import Path
from typing import Optional, List
from utils.logger import SinLogger
from utils.config import ConfigManager
from utils.model_manager import ModelManager
from core.ai.model import SinModel

class SinCLI(cmd.Cmd):
    prompt = "(SinAI) "
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.logger = SinLogger("CLI")
        self.config = ConfigManager(config_path)
        self.model = SinModel()
        self.manager = ModelManager()
        self.session_history = []
        
        self._setup_completions()
        self.logger.info("CLI session started")
    
    def _setup_completions(self):
        """Настройка автодополнения команд"""
        self.commands = {
            "chat": self.do_chat,
            "train": self.do_train,
            "history": self.do_history,
            "save": self.do_save,
            "load": self.do_load,
            "list": self.do_list,
            "modify": self.do_modify,
            "exit": self.do_exit
        }
    
    def do_chat(self, arg: str) -> None:
        """Начать диалог: chat [сообщение]"""
        if not arg:
            print("Укажите сообщение")
            return
        
        try:
            response = self.model.generate(arg)
            print(f"SinAI: {response}")
            self.session_history.append((arg, response))
        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            print("Произошла ошибка")
    
    def do_train(self, arg: str) -> None:
        """Обучить на файле: train [путь] [--epochs N]"""
        parser = argparse.ArgumentParser()
        parser.add_argument("path", type=str)
        parser.add_argument("--epochs", type=int, default=1)
        args = parser.parse_args(arg.split())
        
        if not Path(args.path).exists():
            print(f"Файл не найден: {args.path}")
            return
        
        try:
            for epoch in range(args.epochs):
                with open(args.path) as f:
                    content = f.read()
                
                data_type = "code" if args.path.endswith(".py") else "text"
                result = self.model.learn(content, data_type)
                
                if result["status"] != "success":
                    print(f"Ошибка: {result.get('message', 'Unknown error')}")
                    break
                
                print(f"Эпоха {epoch + 1}: успешно")
            
            version = self.manager.save(self.model)
            print(f"Обучение завершено! Версия: {version}")
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            print("Ошибка при обучении")
    
    def do_history(self, arg: str) -> None:
        """Показать историю: history [--full]"""
        show_full = "--full" in arg
        
        print("\nИстория сессии:")
        for i, (prompt, response) in enumerate(self.session_history, 1):
            if show_full:
                print(f"{i}. Вы: {prompt}")
                print(f"   SinAI: {response}\n")
            else:
                print(f"{i}. {prompt[:50]}... → {response[:50]}...")
    
    def do_save(self, arg: str) -> None:
        """Сохранить модель: save [описание]"""
        try:
            metadata = {"description": arg} if arg else {}
            version = self.manager.save(self.model, metadata)
            print(f"Модель сохранена как версия {version}")
        except Exception as e:
            print(f"Ошибка при сохранении: {str(e)}")
    
    def do_load(self, arg: str) -> None:
        """Загрузить модель: load [версия]"""
        if not arg:
            print("Укажите версию модели")
            return
        
        try:
            self.model = self.manager.load(arg)
            print(f"Модель версии {arg} загружена")
        except Exception as e:
            print(f"Ошибка при загрузке: {str(e)}")
    
    def do_list(self, arg: str) -> None:
        """Список моделей: list [--detailed]"""
        models = self.manager.list_models()
        
        if not models:
            print("Нет сохраненных моделей")
            return
        
        print("\nДоступные модели:")
        for version, meta in models.items():
            if "--detailed" in arg:
                print(f"\nВерсия: {version}")
                for k, v in meta.items():
                    print(f"  {k}: {v}")
            else:
                print(f"- {version}: {meta.get('description', 'Без описания')}")
    
    def do_modify(self, arg: str) -> None:
        """Модификация кода: modify [файл] [--apply CHANGE_ID]"""
        parser = argparse.ArgumentParser()
        parser.add_argument("file", nargs="?", type=str)
        parser.add_argument("--apply", type=str)
        args = parser.parse_args(arg.split())
        
        if args.apply:
            if self.model.apply_change(args.apply):
                print("Изменение успешно применено")
            else:
                print("Не удалось применить изменение")
            return
        
        if not args.file:
            print("Укажите файл с изменениями")
            return
        
        try:
            with open(args.file) as f:
                new_code = f.read()
            
            result = self.model.propose_change(new_code, f"Изменение из {args.file}")
            
            if result["status"] == "requires_approval":
                print("\nПредложено изменение (ID:", result["change_id"], ")")
                print("Описание:", result["description"])
                print("\nDiff:")
                print("\n".join(result["diff"][:20]))
                print("...\n")
                print("Для применения используйте: modify --apply", result["change_id"])
            else:
                print("Изменение отклонено:", result.get("reason", ""))
                if "details" in result:
                    print("Детали:", result["details"])
        except Exception as e:
            print(f"Ошибка: {str(e)}")
    
    def do_exit(self, arg: str) -> bool:
        """Выйти из интерфейса"""
        print("Завершение работы...")
        self.logger.info("CLI session ended")
        return True
    
    def precmd(self, line: str) -> str:
        self.logger.debug(f"Command: {line}")
        return line
    
    def postcmd(self, stop: bool, line: str) -> bool:
        return stop

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    SinCLI(config_path).cmdloop()
