
import json
import random
from uuid import uuid4
from datetime import datetime

# Категории и подкатегории
CATEGORIES = {
    "Технологии": ["смартфоны", "ноутбуки", "искусственный интеллект", "гаджеты", "программирование"],
    "Наука": ["физика", "химия", "биология", "астрономия", "математика"],
    "Развлечения": ["фильмы", "музыка", "игры", "книги", "знаменитости"],
    "Повседневная жизнь": ["кулинария", "шоппинг", "путешествия", "здоровье", "фитнес"],
    "Автомобили": ["электромобили", "тюнинг", "гоночные авто", "ремонт", "марки машин"],
    "Спорт": ["футбол", "хоккей", "теннис", "баскетбол", "бокс"]
}

EMOTIONS = ["нейтральный", "игривый", "образовательный", "смешной", "серьёзный", "дружелюбный"]
LEVELS = ["лёгкий", "средний", "сложный"]

PHRASES = {
    "вопросы": [
        "Что ты думаешь о {topic}?",
        "Как тебе {topic}?",
        "Расскажи про {topic}",
        "Почему {topic} так популярен?",
        "Объясни {topic} простыми словами"
    ],
    "ответы": [
        "Я считаю, что {topic} — это {opinion}",
        "Если честно, {topic} мне {opinion}",
        "На мой взгляд, {topic} {opinion}"
    ],
    "мнения": [
        "очень интересная тема",
        "довольно сложный вопрос",
        "заслуживает внимания"
    ]
}

def generate_question(topic):
    return random.choice(PHRASES["вопросы"]).format(topic=topic)

def generate_answer(topic):
    opinion = random.choice(PHRASES["мнения"])
    return random.choice(PHRASES["ответы"]).format(topic=topic, opinion=opinion)

def generate_responses(topic, category):
    responses = []
    jargon = {
        "Технологии": ["алгоритмы", "интерфейс", "оптимизация"],
        "Наука": ["гипотеза", "эксперимент", "теория"],
        "Экономика": ["инвестиции", "ликвидность", "рынок"]
    }
    
    for _ in range(random.randint(1, 3)):
        response = {
            "text": generate_answer(topic),
            "meta": {
                "difficulty": random.choice(LEVELS),
                "emotion": random.choice(EMOTIONS),
                "slang": random.random() > 0.9,
                "category": category  # Явно добавляем категорию
            }
        }
        
        # Добавляем жаргон только для сложных ответов
        if response["meta"]["difficulty"] in ["сложный"] and category in jargon:
            response["text"] += f" {random.choice(jargon[category])} играет важную роль."
            
        responses.append(response)
    return responses

def generate_dialogue():
    category = random.choice(list(CATEGORIES.keys()))
    subcategory = random.choice(CATEGORIES[category])
    
    return {
        "category": category,
        "subcategory": subcategory,
        "user_query": generate_question(subcategory),
        "responses": generate_responses(subcategory, category)  # Явно передаем категорию
    }

def generate_dataset(num_dialogues=100):
    return {
        "dialogues": [generate_dialogue() for _ in range(num_dialogues)],
        "metadata": {
            "version": "3.0",
            "created_at": datetime.now().isoformat(),
            "categories": list(CATEGORIES.keys())
        }
    }

# Генерация и сохранение
dataset = generate_dataset(1000)  # Генерируем 1000 диалогов для теста
with open("russian_dialogues_final.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Успешно создан датасет с {len(dataset['dialogues'])} диалогами")
