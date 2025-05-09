import json
import zlib
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class APICache:
    """Кэш для хранения ответов API с TTL (временем жизни)."""
    
    def __init__(self, cache_dir: str = "data/api_cache", ttl_hours: int = 24, hash_algo: str = "md5"):
        """
        Инициализация кэша.
        
        Args:
            cache_dir: Директория для хранения кэш-файлов
            ttl_hours: Время жизни кэша в часах
            hash_algo: Алгоритм хеширования (md5, sha1, sha256)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.hash_func = getattr(hashlib, hash_algo)
    
    def _get_cache_path(self, key: str) -> Path:
        """Генерация пути к файлу кэша на основе ключа."""
        return self.cache_dir / f"{self.hash_func(key.encode()).hexdigest()}.json"
    
    def _is_expired(self, cache_file: Path) -> bool:
        """Проверяет, истек ли срок действия кэш-файла."""
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return datetime.now() > datetime.fromisoformat(data['expires_at'])
    
    def get(self, key: str) -> Optional[Any]:
        """Получение данных из кэша."""
        try:
            cache_file = self._get_cache_path(key)
            if not cache_file.exists():
                return None
                
            if self._is_expired(cache_file):
                cache_file.unlink()
                return None
                
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['response']
                
        except (json.JSONDecodeError, IOError) as e:
            cache_file.unlink()  # Удаляем битый кэш
            return None
    
    def set(self, key: str, response: Any, compress: bool = False) -> None:
        """Сохранение данных в кэш."""
        cache_file = self._get_cache_path(key)
        data = {
            'response': response,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + self.ttl).isoformat()
        }
        
        try:
            if compress:
                with open(cache_file, 'wb') as f:
                    f.write(zlib.compress(json.dumps(data).encode()))
            else:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
        except IOError as e:
            raise IOError(f"Failed to write cache: {str(e)}")
    
    def clear(self) -> None:
        """Очистка всего кэша."""
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
    
    def get_stats(self) -> Dict[str, int]:
        """Получение статистики кэша."""
        files = list(self.cache_dir.glob("*.json"))
        expired = sum(1 for f in files if self._is_expired(f))
        return {
            'total': len(files),
            'expired': expired,
            'valid': len(files) - expired
        }
