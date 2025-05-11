from sin import Sin
import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
import torch
import json
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Создаем директорию для логов если ее нет
    os.makedirs('data/logs', exist_ok=True)
    
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

def print_help():
    print("\nДоступные команды:")
    print("  /help - показать это сообщение")
    print("  /save [имя] - сохранить модель (с опциональным именем)")
    print("  /models - список доступных моделей")
    print("  /model_info - подробная информация о моделях")
    print("  /load <имя> - загрузить другую модель")
    print("  /reset - очистить историю диалога")
    print("  /train - начать обучение")
    print("  /eval - оценить модель")
    print("  /compare <модель1> <модель2> - сравнить модели")
    print("  /memory - показать текущую память")
    print("  /report - показать отчет о последнем обучении")
    print("  /config - показать конфигурацию модели")
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
            
        elif cmd == "/model_info":
            models_info = ai.get_model_info()
            print("\nИнформация о моделях:")
            for i, info in enumerate(models_info, 1):
                print(f"\n{i}. {info['name']}")
                print(f"  Размер: {info['size'] / 1024 / 1024:.2f} MB")
                print(f"  Изменена: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                if 'metadata' in info:
                    print("  Метаданные:")
                    print(f"    Версия: {info['metadata'].get('version', 'N/A')}")
                    print(f"    Сохранена: {info['metadata'].get('saved_at', 'N/A')}")
            return True
            
        elif cmd == "/load" and len(parts) > 1:
            model_name = parts[1]
            ai.model = ai._load_model(ai.models_dir / model_name)
            print(f"Модель {model_name} загружена")
            return True
            
        elif cmd == "/reset":
            ai.memory.context.clear()
            print("История диалога очищена")
            return True
            
        elif cmd == "/train":
            print("Начинаем обучение...")
            train_log = ai.train(epochs=3)
            print(f"Обучение завершено. Лучшая эпоха: {ai.monitor.get_best_epoch()}")
            return True
            
        elif cmd == "/eval":
            print("Оценка модели...")
            metrics = ai.evaluate(None)  # Можно передать тестовый датасет
            print("Метрики модели:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            return True
            
        elif cmd == "/compare" and len(parts) > 2:
            model1 = parts[1]
            model2 = parts[2]
            print(f"Сравнение моделей {model1} и {model2}...")
            results = ai.compare_models(
                [ai.models_dir / model1, ai.models_dir / model2],
                None  # Можно передать тестовый датасет
            )
            print("Результаты сравнения:")
            for model_name, metrics in results.items():
                print(f"\n{model_name}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
            return True
            
        elif cmd == "/memory":
            print("\nТекущая память:")
            print("Контекст:")
            for i, msg in enumerate(ai.memory.context, 1):
                print(f"  {i}. {msg}")
            return True
            
        elif cmd == "/report":
            report = ai.get_training_report()
            if report:
                print("\nОтчет о последнем обучении:")
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print("Отчет об обучении не найден")
            return True
            
        elif cmd == "/config":
            print("\nКонфигурация модели:")
            print(f"  Размер словаря: {len(ai.model.tokenizer)}")
            print(f"  Параметры модели: {sum(p.numel() for p in ai.model.parameters())}")
            print(f"  CUDA доступно: {torch.cuda.is_available()}")
            print(f"  Устройство модели: {next(ai.model.parameters()).device}")
            return True
            
    except Exception as e:
        print(f"Ошибка выполнения команды: {str(e)}")
        return True
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Sin - Russian AI Assistant")
    parser.add_argument('--train', action='store_true', help="Enable training mode")
    parser.add_argument('--model', type=str, help="Path to specific model to load")
    parser.add_argument('--data', type=str, default='data/conversations', 
                      help="Path to training data directory")
    args = parser.parse_args()
    
    ai = Sin(args.model) if args.model else Sin()
    
    if args.train:
        logger.info(f"Starting training process with data from: {args.data}")
        try:
            if not os.path.exists(args.data):
                logger.error(f"Data directory not found: {args.data}")
                return
                
            train_log = ai.train(data_path=args.data, epochs=3)
            logger.info(f"Training complete | Best epoch: {ai.monitor.get_best_epoch()}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
        return
    
    print("Sin: Привет! Я Sin, твой русскоязычный ИИ помощник.")
    print("     Напиши /help для списка команд или 'выход' чтобы завершить диалог.\n")
    
    while True:
        try:
            user_input = input("Ты: ").strip()
            
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
