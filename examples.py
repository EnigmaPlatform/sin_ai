"""
Примеры использования SinNetwork
"""

from sin_ai.core.network import SinNetwork

def basic_usage():
    # Инициализация
    sin = SinNetwork()
    
    # Простой чат
    response = sin.communicate("Привет! Как дела?")
    print("Ответ:", response)
    
    # Обучение из файла
    sin.learn_from_file("data/sample.txt")
    
    # Обучение на коде
    python_code = """
def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)
"""
    sin.learn_from_code(python_code, "python")
    
    # Запрос к API
    api_response = sin.query_deepseek("Что такое GPT?")
    print("API ответ:", api_response)

if __name__ == "__main__":
    basic_usage()
