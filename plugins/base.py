# plugins/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class SinPlugin(ABC):
    @abstractmethod
    def initialize(self, network: 'SinNetwork') -> None:
        """Инициализация плагина с доступом к основной сети"""
        pass
    
    @abstractmethod
    def get_commands(self) -> Dict[str, str]:
        """Возвращает словарь команд плагина {команда: описание}"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, args: str) -> Any:
        """Выполнение команды плагина"""
        pass
