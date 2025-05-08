# core/sandbox.py

import sys
import io
import contextlib
import traceback
import re
import resource
from typing import Dict, Any, Optional
from multiprocessing import Process, Queue
import logging

logger = logging.getLogger(__name__)


class CodeSandbox:
    def __init__(self):
        self.safe_modules = {
            'math', 'datetime', 'collections', 'itertools',
            'numpy', 'pandas', 'json', 're'
        }
        self.max_execution_time = 5  # seconds
        self.max_memory_usage = 100  # MB

    def test_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """Безопасное выполнение кода с ограничениями"""
        # Добавляем проверки на опасные конструкции
        forbidden_patterns = [
            r'__.*__', r'os\.', r'sys\.', r'subprocess\.',
            r'open\(', r'eval\(', r'exec\(', r'import\s+os',
            r'import\s+sys', r'import\s+subprocess'
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, code):
                return {
                    'success': False,
                    'error': f"Запрещенная конструкция: {pattern}"
                }

        # Создаем безопасное окружение
        safe_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name)
                for name in dir(__builtins__)
                if name in self._get_safe_builtins()
            }
        }

        # Добавляем безопасные модули
        for mod in self.safe_modules:
            try:
                safe_globals[mod] = __import__(mod)
            except ImportError:
                logger.warning(f"Module {mod} not available")

        # Ограничиваем ресурсы
        def set_limits():
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.max_execution_time, self.max_execution_time)
            )
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.max_memory_usage * 1024 * 1024, self.max_memory_usage * 1024 * 1024)
            )

        # Выполнение с ограничениями
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            result_queue = Queue()

            def worker():
                try:
                    set_limits()
                    local_vars = {}
                    exec(code, safe_globals, local_vars)
                    result_queue.put({
                        'success': True,
                        'variables': {
                            k: v for k, v in local_vars.items()
                            if not k.startswith('__')
                        },
                        'output': captured_output.getvalue()
                    })
                except Exception as e:
                    result_queue.put({
                        'success': False,
                        'error': traceback.format_exc(),
                        'output': captured_output.getvalue()
                    })

            p = Process(target=worker)
            p.start()
            p.join(timeout=self.max_execution_time + 1)

            if p.is_alive():
                p.terminate()
                p.join()
                raise TimeoutError("Execution timed out")

            if result_queue.empty():
                raise RuntimeError("No result from worker process")

            return result_queue.get()

        except Exception as e:
            return {
                'success': False,
                'error': traceback.format_exc(),
                'output': captured_output.getvalue()
            }
        finally:
            sys.stdout = old_stdout

    def _get_safe_builtins(self) -> set:
        return {
            'range', 'enumerate', 'len', 'list', 'dict', 'set', 'tuple',
            'str', 'int', 'float', 'bool', 'max', 'min', 'sum', 'abs',
            'zip', 'sorted', 'reversed', 'isinstance', 'type', 'round'
        }
