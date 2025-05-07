# Управление версиями
import os
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
from utils.logger import SinLogger

class EnhancedModelManager:
    def __init__(self, models_dir: str = "data/models", training_dir: str = "data/training"):
        self.models_dir = Path(models_dir)
        self.training_dir = Path(training_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = SinLogger("ModelManager")
        self.version_history = []
        self._load_version_history()

        # Инициализация компонентов
        self.tokenizer = AutoTokenizer.from_pretrained("DeepSeek/ai-base")
        self.experience = defaultdict(int)
        self.feedback_log = []
        self.model_hashes = set()

    def save_model(self, model, metadata: Optional[Dict] = None) -> str:
        """Улучшенное сохранение модели с проверкой целостности"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"model_{version}.pt"
        
        try:
            # Сохранение модели
            model_data = {
                'state_dict': model.state_dict(),
                'embeddings': model.get_embeddings(),
                'experience': dict(self.experience),
                'config': model.config
            }
            torch.save(model_data, model_path)
            
            # Проверка целостности
            model_hash = self._calculate_file_hash(model_path)
            self.model_hashes.add(model_hash)
            
            # Метаданные
            meta = {
                'version': version,
                'hash': model_hash,
                'skills': self._assess_skills(model),
                'training_data': self._get_training_stats(),
                'feedback_score': self._calculate_feedback_score(),
                'size': os.path.getsize(model_path),
                **metadata
            }
            
            with open(f"{model_path}.meta", 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Обновление истории
            self.version_history.append({
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'metrics': meta['skills']
            })
            self._save_version_history()
            
            # Очистка старых версий
            self._cleanup_old_models(max_keep=5)
            
            self.logger.info(f"Model saved: {version} (hash: {model_hash[:8]}...)")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            if model_path.exists():
                model_path.unlink()
            raise

    def load_model(self, version: str, device: str = "cpu") -> SinNetwork:
        """Безопасная загрузка модели с проверкой"""
        model_path = self.models_dir / f"model_{version}.pt"
        meta_path = self.models_dir / f"model_{version}.pt.meta"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Проверка хэша
            current_hash = self._calculate_file_hash(model_path)
            with open(meta_path) as f:
                meta = json.load(f)
                
            if meta.get('hash') != current_hash:
                raise ValueError("Model file integrity check failed")
            
            # Загрузка данных
            data = torch.load(model_path, map_location=device)
            
            # Создание и настройка модели
            model = SinNetwork(config=data.get('config'))
            model.load_state_dict(data['state_dict'])
            model.load_embeddings(data['embeddings'])
            
            # Восстановление состояния
            self.experience.update(data['experience'])
            
            self.logger.info(f"Model loaded: {version}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {version}: {str(e)}")
            raise

    def verify_model(self, version: str) -> bool:
        """Проверка целостности модели"""
        model_path = self.models_dir / f"model_{version}.pt"
        meta_path = self.models_dir / f"model_{version}.pt.meta"
        
        if not model_path.exists() or not meta_path.exists():
            return False
            
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            current_hash = self._calculate_file_hash(model_path)
            return meta.get('hash') == current_hash
        except:
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Вычисление SHA-256 хэша файла"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_version_history(self) -> None:
        """Загрузка истории версий"""
        history_file = self.models_dir / "versions.json"
        if history_file.exists():
            with open(history_file) as f:
                self.version_history = json.load(f)

    def _save_version_history(self) -> None:
        """Сохранение истории версий"""
        history_file = self.models_dir / "versions.json"
        with open(history_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)

    # Остальные методы остаются аналогичными, но с добавлением логирования
    # и проверок целостности
