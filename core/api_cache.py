import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from typing import Dict, Any

class APICache:
    def __init__(self, cache_dir: str = "data/api_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    
    def get(self, key: str) -> Any:
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None
            
        with open(cache_file, 'r') as f:
            data = json.load(f)
            
        if datetime.now() > datetime.fromisoformat(data['expires_at']):
            cache_file.unlink()
            return None
            
        return data['response']
    
    def set(self, key: str, response: Any) -> None:
        cache_file = self._get_cache_path(key)
        data = {
            'response': response,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + self.ttl).isoformat()
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)
