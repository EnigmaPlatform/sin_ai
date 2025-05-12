import json
import random
from uuid import uuid4

# Константы для генерации данных
CATEGORIES = {
    "technology": ["smartphones", "laptops", "AI", "gadgets", "programming"],
    "science": ["physics", "chemistry", "biology", "astronomy", "mathematics"],
    "entertainment": ["movies", "music", "games", "books", "celebrities"],
    "daily_life": ["cooking", "shopping", "travel", "health", "fitness"]
}

EMOTIONS = ["neutral", "playful", "educational", "funny", "serious", "friendly"]
DIFFICULTIES = ["easy", "medium", "hard"]

def generate_response():
    """Генерация вариантов ответа с метаданными"""
    responses = []
    for _ in range(random.randint(1, 3)):  # 1-3 варианта ответа
        response = {
            "text": generate_text_response(),
            "meta": {
                "difficulty": random.choice(DIFFICULTIES),
                "emotion": random.choice(EMOTIONS),
                "slang": random.random() > 0.7  # 30% вероятность сленга
            }
        }
        responses.append(response)
    return responses

def generate_text_response():
    """Генерация текста ответа"""
    templates = [
        "Это интересный вопрос. {answer}",
        "Я думаю, что {answer}",
        "Насколько мне известно, {answer}",
        "{answer}, если я правильно понимаю.",
        "Отличный вопрос! {answer}"
    ]
    answers = [
        "ответ зависит от многих факторов",
        "можно рассмотреть это с разных точек зрения",
        "существует несколько подходов к этому вопросу",
        "современные исследования показывают интересные результаты",
        "это популярная тема для обсуждения"
    ]
    return random.choice(templates).format(answer=random.choice(answers))

def generate_user_query(category, subcategory):
    """Генерация пользовательского запроса"""
    queries = [
        f"Что ты знаешь о {subcategory}?",
        f"Расскажи мне про {subcategory}",
        f"Как ты относишься к {subcategory}?",
        f"Объясни {subcategory} как для новичка",
        f"Давай обсудим {subcategory}",
        f"Почему {subcategory} так популярен в {category}?"
    ]
    return random.choice(queries)

def generate_dialogue():
    """Генерация одного диалога"""
    category = random.choice(list(CATEGORIES.keys()))
    subcategory = random.choice(CATEGORIES[category])
    
    dialogue = {
        "category": category,
        "subcategory": subcategory,
        "user_query": generate_user_query(category, subcategory),
        "responses": generate_response()
    }
    return dialogue

def generate_dataset(num_dialogues=10000):
    """Генерация всего датасета"""
    dataset = {
        "dialogues": [generate_dialogue() for _ in range(num_dialogues)]
    }
    return dataset

# Генерация и сохранение датасета
dataset = generate_dataset()
with open("dialogues_dataset_schema.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Датасет из {len(dataset['dialogues'])} диалогов успешно создан")
