from sin import Sin
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Sin - Russian AI Assistant")
    parser.add_argument('--train', action='store_true', help="Enable training mode")
    args = parser.parse_args()
    
    ai = Sin()
    
    if args.train:
        print("🔧 Starting training process...")
        try:
            loss = ai.train()
            print(f"✅ Training complete | Final loss: {loss:.4f}")
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
        return
    
    # Режим чата
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
            sys.exit(0)
        except Exception as e:
            print(f"Sin: Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()
