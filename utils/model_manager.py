# Управление версиями
import os
import json
from datetime import datetime
import hashlib

class ModelManager:
    def __init__(self, models_dir="data/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(self, model, metadata=None):
        """Сохраняет модель с автоматической версионизацией"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"model_{version}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Сохраняем метаданные
        meta = {
            'version': version,
            'hash': self._calculate_hash(model_path),
            'created': str(datetime.now()),
            **metadata
        }
        with open(f"{model_path}.meta", 'w') as f:
            json.dump(meta, f)
        
        self._cleanup_old_models(max_keep=5)
        return version

    def load_model(self, version):
        """Загружает конкретную версию модели"""
        model_path = os.path.join(self.models_dir, f"model_{version}.pt")
        model = SinNetwork()  # Ваша архитектура модели
        model.load_state_dict(torch.load(model_path))
        return model

    def _calculate_hash(self, file_path):
        """Вычисляет хеш файла модели"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _cleanup_old_models(self, max_keep=5):
        """Удаляет старые модели, сохраняя только max_keep"""
        models = sorted(os.listdir(self.models_dir))
        while len(models) > max_keep:
            os.remove(os.path.join(self.models_dir, models.pop(0)))
