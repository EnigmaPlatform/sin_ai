import os
import shutil
import torch

def manage_models(models_dir, max_models=5):
    """Управление версиями моделей"""
    models = []
    for f in os.listdir(models_dir):
        if f.startswith('sin_model') and f.endswith('.pt'):
            models.append(os.path.join(models_dir, f))
    
    if len(models) > max_models:
        models.sort(key=os.path.getmtime)
        for old_model in models[:-max_models]:
            os.remove(old_model)

def prepare_device():
    """Определение доступного устройства"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_custom_json(filepath):
    schema = {
        "type": "object",
        "properties": {
            "dialogues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "subcategory": {"type": "string"},
                        "user_query": {"type": "string"},
                        "responses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "meta": {
                                        "type": "object",
                                        "properties": {
                                            "difficulty": {
                                                "type": "string",
                                                "enum": ["easy", "medium", "hard"]
                                            },
                                            "emotion": {
                                                "type": "string",
                                                "enum": ["neutral", "playful", "educational", 
                                                        "funny", "serious", "friendly"]
                                            },
                                            "slang": {"type": "boolean"}
                                        },
                                        "required": ["difficulty", "emotion"]
                                    }
                                },
                                "required": ["text", "meta"]
                            }
                        }
                    },
                    "required": ["category", "user_query", "responses"]
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "total_dialogues": {"type": "number"},
                    "categories_distribution": {"type": "object"},
                    "slang_usage": {"type": "string"},
                    "emotions_distribution": {"type": "object"}
                }
            }
        },
        "required": ["dialogues"]
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        validate(instance=data, schema=schema)
        return True
    except Exception as e:
        print(f"Invalid JSON format in {filepath}: {str(e)}")
        return False

# Инициализация
ai = Sin()

# Обучение на новом JSON
ai.train()

# Использование метаданных в генерации ответа
def generate_response_with_style(user_input, emotion="neutral"):
    context = ai.memory.get_context()
    prompt = f"[{emotion}] {context}\nUser: {user_input}\nSin:"
    return ai.model.generate_response(prompt)
    
def detect_data_level(data):
    if not data.get('dialogues'):
        return "invalid"
    
    first_dialogue = data['dialogues'][0]
    
    if 'meta' in first_dialogue.get('responses', [{}])[0]:
        return "full_meta"
    elif 'category' in first_dialogue:
        return "with_categories"
    else:
        return "basic"
