from sin import Sin
import argparse
import logging

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
