import importlib
import logging
from pathlib import Path
from typing import Dict, Type, List, Any
from sin_ai.core.network import SinNetwork
from sin_ai.plugins.base import SinPlugin

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Менеджер для загрузки и управления плагинами Sin AI.
    
    Пример использования:
    >>> plugin_manager = PluginManager(sin_instance)
    >>> plugins = plugin_manager.load_plugins()
    >>> for plugin in plugins.values():
    ...     plugin.initialize(sin_instance)
    """
    
    def __init__(self, sin_instance: SinNetwork):
        self.sin = sin_instance
        self.plugins_dir = Path(__file__).parent.parent / "plugins"
        self.loaded_plugins: Dict[str, SinPlugin] = {}

    def discover_plugins(self) -> List[Path]:
        """Поиск доступных плагинов в директории"""
        if not self.plugins_dir.exists():
            logger.warning(f"Директория плагинов не найдена: {self.plugins_dir}")
            return []
            
        return list(self.plugins_dir.glob("*.py"))

    def load_plugin(self, plugin_file: Path) -> Type[SinPlugin] | None:
        """Загрузка одного плагина из файла"""
        if plugin_file.name.startswith('_') or plugin_file.name == 'base.py':
            return None
            
        module_name = f"sin_ai.plugins.{plugin_file.stem}"
        
        try:
            module = importlib.import_module(module_name)
            
            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and 
                    issubclass(obj, SinPlugin) and 
                    obj != SinPlugin):
                    logger.info(f"Найден плагин: {name} в {plugin_file.name}")
                    return obj
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки плагина {plugin_file}: {str(e)}")
            
        return None

    def initialize_plugin(self, plugin_class: Type[SinPlugin]) -> SinPlugin | None:
        """Инициализация экземпляра плагина"""
        try:
            plugin = plugin_class()
            plugin.initialize(self.sin)
            return plugin
        except Exception as e:
            logger.error(f"Ошибка инициализации плагина: {str(e)}")
            return None

    def load_plugins(self) -> Dict[str, SinPlugin]:
        """Загрузка и инициализация всех доступных плагинов"""
        self.loaded_plugins = {}
        
        for plugin_file in self.discover_plugins():
            plugin_class = self.load_plugin(plugin_file)
            if plugin_class:
                plugin_instance = self.initialize_plugin(plugin_class)
                if plugin_instance:
                    self.loaded_plugins[plugin_file.stem] = plugin_instance
                    
        logger.info(f"Успешно загружено {len(self.loaded_plugins)} плагинов")
        return self.loaded_plugins

    def get_plugin(self, name: str) -> SinPlugin | None:
        """Получение загруженного плагина по имени"""
        return self.loaded_plugins.get(name)

    def get_available_commands(self) -> Dict[str, str]:
        """Получение списка всех команд из плагинов"""
        commands = {}
        for plugin in self.loaded_plugins.values():
            commands.update(plugin.get_commands())
        return commands

    def execute_plugin_command(self, command: str, args: str) -> Any:
        """Выполнение команды плагина"""
        for plugin in self.loaded_plugins.values():
            if command in plugin.get_commands():
                try:
                    return plugin.execute_command(command, args)
                except Exception as e:
                    logger.error(f"Ошибка выполнения команды {command}: {str(e)}")
                    return f"Ошибка плагина: {str(e)}"
        return None
