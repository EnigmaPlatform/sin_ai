import torch
import json
from pathlib import Path
from typing import Optional
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_dir: str = "data/learned/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
    
    def save_model(self, model, model_name: str) -> None:
        """Сохранение модели"""
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True, parents=True)
        
        # Сохранение параметров модели
        torch.save(model.state_dict(), model_path / "model_weights.pt")
        
        # Сохранение метаданных
        metadata = {
            'model_name': model_name,
            'save_date': datetime.now().isoformat(),
            'model_class': model.__class__.__name__,
            'parameters': {
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(model.device)
            }
        }
        
        with open(model_path / "metadata.json",
