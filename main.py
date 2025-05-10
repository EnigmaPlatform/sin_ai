from sin import Sin
import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Sin - Russian AI Assistant")
    parser.add_argument('--train', action='store_true', help="Enable training mode")
    args = parser.parse_args()
    
    ai = Sin()
    
    if args.train:
        logger.info("Starting training process...")
        try:
            loss = ai.train()
            logger.info(f"Training complete | Loss: {loss:.4f}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
        return
    
    print("Sin: Привет! Я Sin, твой русскоязычный ИИ помощник.")
    print("     Напиши 'выход' чтобы завершить диалог.\n")
    
    while True:
        try:
            user_input = input("Ты: ").strip()
            
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print("Sin: До новых встреч!")
                ai.save()
                break
                
            response = ai.chat(user_input)
            print(f"Sin: {response}")
            
        except KeyboardInterrupt:
            print("\nSin: Сохраняю данные перед выходом...")
            ai.save()
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print("Sin: Произошла ошибка, попробуйте другой вопрос")

if __name__ == "__main__":
    main()

def handle_command(self, command):
    """Обработка специальных команд"""
    if command == "/save":
        try:
            self.save()
            return "Данные сохранены"
        except Exception as e:
            return f"Ошибка сохранения: {str(e)}"
    elif command == "/reset":
        self.memory.context.clear()
        return "Контекст очищен"
    elif command == "/help":
        return ("Доступные команды:\n"
                "/save - сохранить состояние\n"
                "/reset - очистить историю\n"
                "/help - справка")
    return None

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Логи в файл с ротацией
    file_handler = RotatingFileHandler(
        'data/logs/sin.log',
        maxBytes=5*1024*1024,
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    
    # Логи в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()
