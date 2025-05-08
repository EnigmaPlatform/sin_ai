from functools import wraps
from typing import Callable, Any, Dict
from pathlib import Path
import re

def validate_input(**validators: Dict[str, Callable]) -> Callable:
    """Декоратор для валидации входных параметров"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Получаем имена параметров
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Проверяем каждый параметр
            for param, value in bound.arguments.items():
                if param in validators:
                    validator = validators[param]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param}: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Стандартные валидаторы
def validate_text(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 0

def validate_file_path(path: str) -> bool:
    return isinstance(path, str) and Path(path).exists()

def validate_language(lang: str) -> bool:
    return lang.lower() in {'python', 'javascript', 'java', 'c++', 'go'}

def validate_model_name(name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_-]{3,50}$', name))
