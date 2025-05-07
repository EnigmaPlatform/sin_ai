import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

class SinLogger:
    def __init__(self, name: str = "SinAI", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_console_logger()
        self._setup_file_logger()
        self._setup_audit_logger()

    def _setup_console_logger(self) -> None:
        """Настройка вывода в консоль"""
        self.console_logger = logging.getLogger(f"{self.name}.console")
        self.console_logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.console_logger.addHandler(handler)

    def _setup_file_logger(self) -> None:
        """Настройка ротации лог-файлов"""
        self.file_logger = logging.getLogger(f"{self.name}.file")
        self.file_logger.setLevel(logging.DEBUG)
        
        handler = RotatingFileHandler(
            self.log_dir / "sinai.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.file_logger.addHandler(handler)

    def _setup_audit_logger(self) -> None:
        """Настройка аудит-лога для критических операций"""
        self.audit_logger = logging.getLogger(f"{self.name}.audit")
        self.audit_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / "audit.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.audit_logger.addHandler(handler)

    def log_operation(self, operation: str, metadata: Dict[str, Any]) -> None:
        """Логирование важных операций с метаданными"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "metadata": metadata
        }
        self.audit_logger.info(json.dumps(log_entry))

    def info(self, message: str) -> None:
        self.console_logger.info(message)
        self.file_logger.info(message)

    def warning(self, message: str) -> None:
        self.console_logger.warning(message)
        self.file_logger.warning(message)

    def error(self, message: str) -> None:
        self.console_logger.error(message)
        self.file_logger.error(message)

    def debug(self, message: str) -> None:
        self.file_logger.debug(message)
