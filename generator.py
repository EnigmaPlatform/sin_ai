import json
import random
from datetime import datetime, timedelta
from uuid import uuid4

# Списки для генерации случайных данных
topics = ["технологии", "наука", "искусство", "спорт", "политика", "кино", "музыка", "путешествия", "еда", "здоровье"]
languages = ["ru", "en", "de", "fr", "es", "zh"]
sentiments = ["positive", "neutral", "negative"]
user_types = ["new", "regular", "vip"]
platforms = ["web", "mobile", "desktop"]

def generate_message(speaker, topic):
    """Генерация случайного сообщения"""
    greetings = ["Привет", "Здравствуйте", "Добрый день", "Hi", "Hello"]
    questions = [
        f"Что ты думаешь о {topic}?",
        f"Как тебе {topic}?",
        f"Можешь рассказать о {topic}?",
        f"Почему {topic} так популярен?",
        f"Согласен ли ты, что {topic} это важно?"
    ]
    responses = [
        f"Я считаю, что {topic} это интересно.",
        f"Мне нравится {topic}.",
        f"Я не очень разбираюсь в {topic}.",
        f"{topic.capitalize()} - это актуальная тема.",
        f"Давай обсудим {topic} подробнее."
    ]
    
    if speaker == "user":
        if random.random() > 0.7:
            return random.choice(greetings)
        return random.choice(questions)
    else:
        return random.choice(responses)

def generate_dialog():
    """Генерация одного диалога"""
    topic = random.choice(topics)
    language = random.choice(languages)
    sentiment = random.choice(sentiments)
    user_type = random.choice(user_types)
    platform = random.choice(platforms)
    
    # Генерация случайной даты в последние 30 дней
    date = datetime.now() - timedelta(days=random.randint(0, 30))
    
    dialog_id = str(uuid4())
    num_messages = random.randint(2, 10)  # От 2 до 10 сообщений в диалоге
    
    messages = []
    for i in range(num_messages):
        speaker = "user" if i % 2 == 0 else "assistant"
        message = {
            "message_id": i + 1,
            "speaker": speaker,
            "text": generate_message(speaker, topic),
            "timestamp": (date + timedelta(minutes=i)).isoformat()
        }
        messages.append(message)
    
    metadata = {
        "dialog_id": dialog_id,
        "topic": topic,
        "language": language,
        "sentiment": sentiment,
        "user_type": user_type,
        "platform": platform,
        "created_at": date.isoformat(),
        "message_count": num_messages,
        "duration_seconds": num_messages * random.randint(5, 30)
    }
    
    return {
        "metadata": metadata,
        "messages": messages
    }

def generate_dataset(num_dialogs=10000):
    """Генерация датасета"""
    dataset = []
    for _ in range(num_dialogs):
        dataset.append(generate_dialog())
    
    return dataset

# Генерация датасета
dataset = generate_dataset()

# Сохранение в JSON файл
with open("dialogs_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Датасет из {len(dataset)} диалогов успешно создан и сохранен в dialogs_dataset.json")
