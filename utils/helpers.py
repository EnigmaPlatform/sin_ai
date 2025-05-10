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
