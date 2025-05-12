import json
import random
from uuid import uuid4

# Категории и подкатегории на русском
CATEGORIES = {
    "Технологии": ["смартфоны", "ноутбуки", "искусственный интеллект", "гаджеты", "программирование"],
    "Наука": ["физика", "химия", "биология", "астрономия", "математика"],
    "Развлечения": ["фильмы", "музыка", "игры", "книги", "знаменитости"],
    "Повседневная жизнь": ["кулинария", "шоппинг", "путешествия", "здоровье", "фитнес"],
    "Автомобили": ["электромобили", "тюнинг", "гоночные авто", "ремонт", "марки машин"],
    "Спорт": ["футбол", "хоккей", "теннис", "баскетбол", "бокс"]
}

# Варианты эмоциональной окраски
EMOTIONS = ["нейтральный", "игривый", "образовательный", "смешной", "серьёзный", "дружелюбный"]
LEVELS = ["лёгкий", "средний", "сложный"]

# База данных для генерации естественных фраз
PHRASES = {
    "вопросы": [
        "Что ты думаешь о {topic}?",
        "Как тебе {topic}?",
        "Расскажи про {topic}",
        "Почему {topic} так популярен?",
        "Объясни {topic} простыми словами",
        "В чём особенность {topic}?",
        "Какое твоё мнение о {topic}?",
        "С чего начать изучение {topic}?",
        "Какие есть интересные факты о {topic}?",
        "Как ты относишься к {topic}?"
    ],
    "ответы": [
        "Я считаю, что {topic} — это {opinion}",
        "Если честно, {topic} мне {opinion}",
        "На мой взгляд, {topic} {opinion}",
        "Многие говорят, что {topic} {opinion}",
        "По моему опыту, {topic} {opinion}",
        "Недавно читал, что {topic} {opinion}",
        "Судя по последним данным, {topic} {opinion}",
        "Как специалист скажу — {topic} {opinion}"
    ],
    "мнения": [
        "очень интересная тема",
        "довольно сложный вопрос",
        "заслуживает внимания",
        "не так прост, как кажется",
        "стал популярен не просто так",
        "меня всегда увлекал",
        "вызывает много споров",
        "сильно изменился в последние годы",
        "важен для современного мира",
        "лучше один раз попробовать, чем сто раз услышать"
    ]
}

def generate_question(topic):
    """Генерация естественного вопроса"""
    return random.choice(PHRASES["вопросы"]).format(topic=topic)

def generate_answer(topic):
    """Генерация естественного ответа"""
    opinion = random.choice(PHRASES["мнения"])
    return random.choice(PHRASES["ответы"]).format(topic=topic, opinion=opinion)

def generate_responses(topic):
    """Генерация вариантов ответов с метаданными"""
    responses = []
    for _ in range(random.randint(1, 3)):  # 1-3 варианта ответа
        response = {
            "text": generate_answer(topic),
            "meta": {
                "difficulty": random.choice(LEVELS),
                "emotion": random.choice(EMOTIONS),
                "slang": random.random() > 0.9  # 10% вероятность сленга
            }
        }
        # Добавляем сленг при необходимости
        if response["meta"]["slang"]:
            response["text"] = response["text"].replace("Я считаю", "Я щитаю").replace("мнение", "мнение (ну типа)")
        responses.append(response)
    return responses

def generate_dialogue():
    """Генерация одного диалога"""
    category = random.choice(list(CATEGORIES.keys()))
    subcategory = random.choice(CATEGORIES[category])
    
    return {
        "category": category,
        "subcategory": subcategory,
        "user_query": generate_question(subcategory),
        "responses": generate_responses(subcategory)
    }

def generate_dataset(num_dialogues=10000):
    """Генерация всего датасета"""
    return {
        "dialogues": [generate_dialogue() for _ in range(num_dialogues)]
    }

# Генерация и сохранение
dataset = generate_dataset()
with open("russian_dialogues.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2, ensure_ascii=False)

print(f"Создан русскоязычный датасет с {len(dataset['dialogues'])} диалогами")
