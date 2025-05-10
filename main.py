import logging
from core.network import SinNetwork
from models.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)

def main():
    ai = SinNetwork()
    manager = ModelManager()
    
    while True:
        message = input("You: ")
        if message.lower() == 'exit':
            break
            
        response = ai.communicate(message)
        print(f"AI: {response}")
        
        # Автоматическое обучение на диалоге
        ai.learn_from_text(f"User: {message}\nAI: {response}")

if __name__ == "__main__":
    main()
