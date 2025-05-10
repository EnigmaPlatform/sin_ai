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

            if user_input.startswith('/'):
            if handle_command(ai, user_input):
                continue
                
        if user_input.lower() in ('выход', 'exit', 'quit', '/exit'):
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

def print_help():
    print("\nДоступные команды:")
    print("  /help - показать это сообщение")
    print("  /save [имя] - сохранить модель (с опциональным именем)")
    print("  /models - список доступных моделей")
    print("  /load <имя> - загрузить другую модель")
    print("  /reset - очистить историю диалога")
    print("  /train - начать обучение")
    print("  /exit - выйти из программы\n")

def handle_command(ai, command):
    """Обработка команд пользователя"""
    try:
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            print_help()
            return True
            
        elif cmd == "/save":
            model_name = parts[1] if len(parts) > 1 else None
            save_path = ai.save_model(model_name)
            print(f"Модель сохранена как: {save_path}")
            return True
            
        elif cmd == "/models":
            models = ai.list_models()
            print("\nДоступные модели:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
            return True
            
        elif cmd == "/load" and len(parts) > 1:
            ai.model = ai._load_model(ai.models_dir / parts[1])
            print(f"Модель {parts[1]} загружена")
            return True
            
        elif cmd == "/reset":
            ai.memory.context.clear()
            print("История диалога очищена")
            return True
            
        elif cmd == "/train":
            print("Начинаем обучение...")
            ai.train(epochs=3)
            return True
            
    except Exception as e:
        print(f"Ошибка выполнения команды: {str(e)}")
        return True
        
    return False

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
