# Интерфейс командной строки
import cmd
from core.network.model import SinNetwork
from utils.model_manager import EnhancedModelManager
from utils.logger import SinLogger
from typing import Optional
import argparse
from pathlib import Path

class EnhancedSinCLI(cmd.Cmd):
    prompt = "(SinAI) "
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.logger = SinLogger("CLI")
        self.manager = EnhancedModelManager()
        self.model = self._initialize_model(config_path)
        self.history = []
        self.current_session = []
        
        self.logger.info("CLI initialized")

    def _initialize_model(self, config_path: Optional[str]) -> SinNetwork:
        """Инициализация модели с возможностью загрузки конфига"""
        try:
            if config_path:
                model = SinNetwork.load_config(config_path)
                self.logger.info(f"Model loaded from config: {config_path}")
            else:
                model = SinNetwork()
                self.logger.info("New model initialized with default config")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def do_chat(self, arg: str) -> None:
        """Начать диалог: chat [сообщение]"""
        if not arg:
            print("Укажите сообщение")
            return
            
        self.history.append(("user", arg))
        self.current_session.append(arg)
        
        try:
            response = self.model.generate_response(arg, context=self.current_session[-3:])
            print(f"SinAI: {response}")
            
            self.history.append(("ai", response))
            self.current_session.append(response)
            
        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            print("Произошла ошибка при обработке запроса")

    def do_train(self, arg: str) -> None:
        """Обучить на файле: train [путь] [--epochs N]"""
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("path", type=str)
            parser.add_argument("--epochs", type=int, default=1)
            args = parser.parse_args(arg.split())
            
            if not Path(args.path).exists():
                print(f"Файл не найден: {args.path}")
                return
                
            print(f"Начало обучения на {args.path} ({args.epochs} эпох)...")
            
            for epoch in range(args.epochs):
                result = self.model.learn_from_file(args.path)
                print(f"Эпоха {epoch + 1}: {result['status']}")
                
                if result['status'] == "error":
                    print(f"Ошибка: {result['message']}")
                    break
                    
            version = self.manager.save_model(self.model)
            print(f"Обучение завершено! Модель сохранена как версия {version}")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            print("Произошла ошибка при обучении")

    def do_history(self, arg: str) -> None:
        """Показать историю диалога: history [--session]"""
        if arg == "--session":
            history = self.current_session
        else:
            history = [f"{who}: {msg}" for who, msg in self.history]
            
        for i, item in enumerate(history, 1):
            print(f"{i}. {item}")

    def do_save(self, arg: str) -> None:
        """Сохранить модель: save [описание]"""
        try:
            version = self.manager.save_model(self.model, {
                "description": arg or "Manual save"
            })
            print(f"Модель сохранена как версия {version}")
        except Exception as e:
            print(f"Ошибка при сохранении: {str(e)}")

    def do_load(self, arg: str) -> None:
        """Загрузить модель: load [версия]"""
        try:
            self.model = self.manager.load_model(arg)
            print(f"Модель версии {arg} успешно загружена")
        except Exception as e:
            print(f"Ошибка при загрузке: {str(e)}")

    def do_list(self, arg: str) -> None:
        """Список доступных моделей: list"""
        models = []
        for file in Path(self.manager.models_dir).glob("model_*.pt"):
            version = file.stem.split("_")[1]
            meta_file = file.with_suffix(file.suffix + ".meta")
            
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                models.append((version, meta.get('description', '')))
        
        if models:
            print("Доступные модели:")
            for version, desc in sorted(models, reverse=True):
                print(f"- {version}: {desc}")
        else:
            print("Нет сохраненных моделей")

    def do_exit(self, arg: str) -> bool:
        """Выйти из интерфейса: exit"""
        print("Завершение работы...")
        self.logger.info("CLI session ended")
        return True

    def postcmd(self, stop: bool, line: str) -> bool:
        """Логирование после выполнения команды"""
        self.logger.debug(f"Command executed: {line}")
        return stop

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    EnhancedSinCLI(config_path).cmdloop()
