import os
import json
from jsonschema import validate
import logging
logger = logging.getLogger(__name__)

JSON_SCHEMA = {
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
                                        "difficulty": {"enum": ["easy", "medium", "hard"]},
                                        "emotion": {"enum": ["neutral", "playful", "educational", "funny", "serious", "friendly"]},
                                        "slang": {"type": "boolean"}
                                    }
                                }
                            },
                            "required": ["text"]
                        }
                    }
                },
                "required": ["user_query", "responses"]
            }
        }
    },
    "required": ["dialogues"]
}

def validate_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        validate(instance=data, schema=JSON_SCHEMA)
        return True
    except Exception as e:
        print(f"Invalid JSON format in {filepath}: {str(e)}")
        return False

def manage_models(models_dir, max_models=5):
    models = sorted(
        [f for f in os.listdir(models_dir) if f.startswith('sin_model') and f.endswith('.pt')],
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f))
    )
    
    if len(models) > max_models:
        for old_model in models[:-max_models]:
            os.remove(os.path.join(models_dir, old_model))
