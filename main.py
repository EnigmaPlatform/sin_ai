import logging
import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv
from core.network import SinNetwork
from ui.interface import CommandLineInterface
from core.monitoring import start_monitoring
from core.plugins import PluginManager
from models.model_manager import ModelManager


def configure_logging(log_level=logging.INFO, log_file='sin_ai.log'):
    """Настройка системы логирования."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    # Настройка логов для библиотек
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def load_environment():
    """Загрузка переменных окружения."""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logging.warning(f"Файл .env не найден по пути: {env_path}")


def parse_args():
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Sin AI - Интеллектуальный ассистент')
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Имя модели для загрузки (по умолчанию используется последняя)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Включить режим отладки (повышенный уровень логирования)'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Включить сервер мониторинга Prometheus'
    )
    
    parser.add_argument(
        '--monitor-port',
        type=int,
        default=8001,
        help='Порт для сервера мониторинга'
    )
    
    parser.add_argument(
        '--no-plugins',
        action='store_true',
        help='Отключить загрузку плагинов'
    )
    
    return parser.parse_args()


def initialize_network(model_name=None):
    """Инициализация нейросети с обработкой ошибок."""
    try:
        from models.model_manager import ModelManager  # Ленивый импорт
        from core.network import SinNetwork
        
        model_manager = ModelManager()
        
        if model_name:
            loaded_model = model_manager.load_model(model_name, SinNetwork)
            if loaded_model:
                logging.info(f"Модель {model_name} успешно загружена")
                return loaded_model
            logging.warning(f"Не удалось загрузить модель {model_name}, создается новая")
        
        return SinNetwork()
        
    except Exception as e:
        logging.critical(f"Ошибка инициализации нейросети: {str(e)}")
        sys.exit(1)


def run_cli_interface(sin_instance, load_plugins=True):
    """Запуск CLI интерфейса."""
    try:
        cli = CommandLineInterface(sin_instance)
        
        if load_plugins:
            plugin_manager = PluginManager(sin_instance)
            plugins = plugin_manager.load_plugins()
            
            if plugins:
                logging.info(f"Загружено {len(plugins)} плагинов")
                cli.register_plugins(plugins)
            else:
                logging.warning("Не удалось загрузить плагины")
        
        cli.run()  # Теперь метод существует
    except KeyboardInterrupt:
        logging.info("Завершение работы по запросу пользователя")
    except Exception as e:
        logging.critical(f"Критическая ошибка в интерфейсе: {str(e)}")
        sys.exit(1)

def main():
    """Основная функция запуска приложения."""
    args = parse_args()
    
    # Настройка логирования
    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(log_level)
    
    # Загрузка окружения
    load_environment()
    
    # Запуск мониторинга
    if args.monitor:
        start_monitoring(port=args.monitor_port)
        logging.info(f"Сервер мониторинга запущен на порту {args.monitor_port}")
    
    # Инициализация нейросети
    sin = initialize_network(args.model)
    
    # Запуск интерфейса
    run_cli_interface(sin, not args.no_plugins)


if __name__ == "__main__":
    main()
