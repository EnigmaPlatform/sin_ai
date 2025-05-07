import importlib
from pathlib import Path
from typing import Dict, Type, Any
import logging
from abc import ABC, abstractmethod

class Plugin(ABC):
    """Базовый класс для всех плагинов"""
    def __init__(self, model: Any):
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Основной метод выполнения плагина"""
        pass

class PluginManager:
    def __init__(self, model: Any, plugins_dir: str = "plugins"):
        self.model = model
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Type[Plugin]] = {}
        self.loaded_plugins: Dict[str, Plugin] = {}
        self.logger = logging.getLogger("PluginManager")
        
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Поиск доступных плагинов"""
        for file in self.plugins_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue
                
            try:
                module_name = f"plugins.{file.stem}"
                module = importlib.import_module(module_name)
                
                for name, obj in module.__dict__.items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, Plugin) and 
                        obj != Plugin):
                        self.plugins[name] = obj
                        self.logger.info(f"Discovered plugin: {name}")
                        
            except Exception as e:
                self.logger.error(f"Failed to load plugin {file.name}: {str(e)}")

    def get_plugin(self, name: str) -> Plugin:
        """Получение экземпляра плагина"""
        if name in self.loaded_plugins:
            return self.loaded_plugins[name]
            
        if name in self.plugins:
            try:
                plugin = self.plugins[name](self.model)
                self.loaded_plugins[name] = plugin
                self.logger.info(f"Loaded plugin: {name}")
                return plugin
            except Exception as e:
                self.logger.error(f"Failed to initialize plugin {name}: {str(e)}")
                raise
                
        raise ValueError(f"Plugin {name} not found")

    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """Выполнение плагина"""
        plugin = self.get_plugin(name)
        return plugin.execute(*args, **kwargs)

    def list_plugins(self) -> Dict[str, str]:
        """Список доступных плагинов с описанием"""
        return {
            name: plugin.__doc__ or "No description"
            for name, plugin in self.plugins.items()
        }
