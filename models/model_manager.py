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
        
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str, model_class) -> Optional:
        """Загрузка модели"""
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            logger.error(f"Model {model_name} not found")
            return None
        
        try:
            # Загрузка модели
            model = model_class()
            model.load_state_dict(torch.load(model_path / "model_weights.pt"))
            model.eval()
            
            logger.info(f"Model {model_name} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def delete_model(self, model_name: str) -> bool:
        """Удаление модели"""
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            logger.error(f"Model {model_name} not found")
            return False
        
        try:
            shutil.rmtree(model_path)
            logger.info(f"Model {model_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def list_models(self) -> List[Dict]:
        """Список доступных моделей"""
        models = []
        
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            models.append(metadata)
                    except Exception as e:
                        logger.error(f"Failed to read metadata for {model_dir.name}: {e}")
        
        return sorted(models, key=lambda x: x.get('save_date', ''), reverse=True)
