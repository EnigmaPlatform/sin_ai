from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.network import SinNetwork

class SinPlugin:
    def initialize(self, network: 'SinNetwork') -> None:
        """Инициализация плагина с доступом к основной сети"""
        self.network = network
    
    def get_commands(self) -> Dict[str, str]:
        """Возвращает словарь команд плагина {команда: описание}"""
        raise NotImplementedError
    
    def execute_command(self, command: str, args: str) -> Any:
        """Выполнение команды плагина"""
        raise NotImplementedError
