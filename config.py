import json
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        
        if self.config_path.exists():
            self._load()
        else:
            self._save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
        self._save()

    def _load(self) -> None:
        with open(self.config_path) as f:
            self.config = {**self.config, **json.load(f)}

    def _save(self) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _load_default_config(self) -> Dict:
        return {
            "model": {
                "name": "DeepSeek/ai-base",
                "max_length": 512,
                "temperature": 0.7
            },
            "memory": {
                "short_term_size": 10,
                "long_term_size": 1000
            },
            "learning": {
                "batch_size": 8,
                "learning_rate": 1e-5,
                "max_files": 100
            },
            "api": {
                "deepseek_key": "",
                "timeout": 10
            },
            "safety": {
                "max_code_size": 64 * 1024 * 1024,
                "allowed_imports": ["math", "numpy"]
            }
        }
