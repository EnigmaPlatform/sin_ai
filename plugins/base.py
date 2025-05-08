from abc import ABC, abstractmethod
from typing import Dict, Any
from sin_ai.core.network import SinNetwork

class SinPlugin(ABC):
    """
    Базовый класс для всех плагинов Sin AI.
    
    Пример реализации плагина:
    >>> class MyPlugin(SinPlugin):
    ...     def initialize(self, network):
    ...         self.network = network
    ...     
    ...     def get_commands(self):
    ...         return {'greet': "Приветствие пользователя"}
    ...     
    ...     def execute_command(self, command, args):
    ...         if command == 'greet':
    ...             return f"Привет, {args}!" if args else "Привет!"
    """

    @abstractmethod
    def initialize(self, network: SinNetwork) -> None:
        """Инициализация плагина с доступом к основной сети"""
        pass
    
    @abstractmethod
    def get_commands(self) -> Dict[str, str]:
        """Возвращает словарь команд плагина {команда: описание}"""
        return {}
    
    @abstractmethod
    def execute_command(self, command: str, args: str) -> Any:
        """Выполнение команды плагина"""
        pass
