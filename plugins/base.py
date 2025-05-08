from abc import ABC, abstractmethod
from typing import Dict, Any

class SinPlugin(ABC):
    @abstractmethod
    def initialize(self, network: Any) -> None:
        """Инициализация плагина"""
        pass
    
    @abstractmethod
    def get_commands(self) -> Dict[str, str]:
        """Возвращает команды для CLI"""
        return {}
    
    @abstractmethod
    def execute_command(self, command: str, args: str) -> Any:
        """Выполнение команды"""
        pass
