from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
from typing import Callable, Any

# Метрики
REQUESTS = Counter('sin_requests_total', 'Total API requests')
REQUEST_LATENCY = Histogram('sin_request_latency_seconds', 'Request latency')
ERRORS = Counter('sin_errors_total', 'Total errors')
LEARNING_PROGRESS = Gauge('sin_learning_progress', 'Learning progress')
MODEL_LEVEL = Gauge('sin_model_level', 'Current model level')


def monitor(fn: Callable) -> Callable:
    """Декоратор для мониторинга функций.
    
    Args:
        fn: Функция, которую нужно мониторить.
        
    Returns:
        Обернутая функция с мониторингом.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        REQUESTS.inc()
        
        try:
            result = fn(*args, **kwargs)
            duration = time.time() - start_time
            REQUEST_LATENCY.observe(duration)
            return result
        except Exception as e:
            ERRORS.inc()
            raise e
    
    return wrapper


def start_monitoring(port: int = 8001) -> None:
    """Запуск сервера мониторинга.
    
    Args:
        port: Порт, на котором будет запущен сервер мониторинга.
    """
    start_http_server(port)
