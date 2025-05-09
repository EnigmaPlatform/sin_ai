"""
Основные модули Sin AI

Импорты организованы таким образом, чтобы избежать циклических зависимостей.
Для типизации используются протоколы из core.types.
"""

# Основной экспорт
from .network import SinNetwork
from typing import Dict, List  # в начале файла
from .types import (
    SinNetworkProtocol,
    MemorySystemProtocol,
    LearningEngineProtocol,
    PluginManagerProtocol
)

# Конкретные реализации (для внутреннего использования)
from .memory import MemorySystem as _MemorySystem
from .learning import LearningEngine as _LearningEngine
from .api_handler import DeepSeekAPIHandler as _DeepSeekAPIHandler
from .code_analyzer import CodeAnalyzer as _CodeAnalyzer
from .sandbox import CodeSandbox as _CodeSandbox
from .level_system import LevelSystem as _LevelSystem
from .plugins import PluginManager as _PluginManager

# Утилиты и сервисы
from .monitoring import start_monitoring, monitor
from .personality import PersonalityCore as _PersonalityCore
from .emotions import EmotionEngine as _EmotionEngine
from .deepseek_trainer import DeepSeekTrainer as _DeepSeekTrainer

# Реэкспорт типов для удобства
__all__ = [
    'SinNetwork',
    'SinNetworkProtocol',
    'MemorySystemProtocol',
    'LearningEngineProtocol',
    'PluginManagerProtocol',
    'start_monitoring',
    'monitor'
]

# Алиасы для внутреннего использования (чтобы избежать циклических импортов)
MemorySystem: MemorySystemProtocol = _MemorySystem
LearningEngine: LearningEngineProtocol = _LearningEngine
DeepSeekAPIHandler = _DeepSeekAPIHandler
CodeAnalyzer = _CodeAnalyzer
CodeSandbox = _CodeSandbox
LevelSystem = _LevelSystem
PluginManager: PluginManagerProtocol = _PluginManager
PersonalityCore = _PersonalityCore
EmotionEngine = _EmotionEngine
DeepSeekTrainer = _DeepSeekTrainer
