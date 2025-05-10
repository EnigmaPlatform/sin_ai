import logging
from core.network import SinNetwork
from models.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    ai = SinNetwork()
    manager = ModelManager()
    
    print("Sin AI - Простой чат-ассистент")
    print("Введите 'save имя' для сохранения, 'load имя' для загрузки, 'exit' для выхода")
    
    while True:
        message = input("Вы: ").strip()
        
        if message.lower() == 'exit':
            break
        elif message.startswith('save '):
            name = message[5:]
            manager.save_model(ai, name)
            print(f"Модель сохранена как {name}")
        elif message.startswith('load '):
            name = message[5:]
            ai = manager.load_model(name, SinNetwork)
            print(f"Модель {name} загружена")
        else:
            response = ai.communicate(message)
            print(f"AI: {response}")
            ai.learn_from_text(f"User: {message}\nAI: {response}")

if __name__ == "__main__":
    main()
