import shutil
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List
import hashlib
import tarfile
import tempfile

class BackupSystem:
    def __init__(self, config: dict):
        self.backup_dir = Path(config['backup_dir'])
        self.max_backups = config.get('max_backups', 5)
        self.logger = logging.getLogger("BackupSystem")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, components: List[str]) -> str:
        """Создание бэкапа указанных компонентов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"sin_backup_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Копирование компонентов во временную директорию
            for component in components:
                src = Path(component)
                if src.exists():
                    dst = temp_path / src.name
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            
            # Создание архива
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(temp_dir, arcname="")
            
        # Проверка целостности
        if not self._verify_backup(backup_path):
            backup_path.unlink()
            raise RuntimeError("Backup verification failed")
            
        self._cleanup_old_backups()
        return str(backup_path)

    def restore_backup(self, backup_file: str, target_dir: str) -> None:
        """Восстановление из бэкапа"""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            raise FileNotFoundError("Backup file not found")
            
        if not self._verify_backup(backup_path):
            raise ValueError("Backup verification failed")
            
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(target)

    def _verify_backup(self, backup_path: Path) -> bool:
        """Проверка целостности бэкапа"""
        try:
            # Проверка структуры архива
            with tarfile.open(backup_path, "r:gz") as tar:
                members = tar.getmembers()
                if not members:
                    return False
                    
            # Проверка хэша
            expected_hash = self._calculate_file_hash(backup_path)
            return True
        except:
            return False

    def _cleanup_old_backups(self) -> None:
        """Удаление старых бэкапов"""
        backups = sorted(self.backup_dir.glob("sin_backup_*.tar.gz"))
        if len(backups) > self.max_backups:
            for old_backup in backups[:-self.max_backups]:
                old_backup.unlink()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Вычисление SHA-256 хэша файла"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def list_backups(self) -> List[dict]:
        """Список доступных бэкапов"""
        backups = []
        for backup in self.backup_dir.glob("sin_backup_*.tar.gz"):
            backups.append({
                "name": backup.name,
                "size": backup.stat().st_size,
                "modified": datetime.fromtimestamp(backup.stat().st_mtime).isoformat()
            })
        return sorted(backups, key=lambda x: x['modified'], reverse=True)
