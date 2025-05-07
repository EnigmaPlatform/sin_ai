import sys
import io
import contextlib
import traceback
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CodeSandbox:
    def __init__(self):
        self.safe_builtins = {
            'range', 'enumerate', 'len', 'list', 'dict', 'set', 'tuple',
            'str', 'int', 'float', 'bool', 'max', 'min', 'sum', 'abs',
            'zip', 'sorted', 'reversed'
        }
    
    def test_code(self, code: str, timeout: int = 5) -> Dict:
        """Безопасное выполнение кода в песочнице"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'variables': {}
        }
        
        # Создаем безопасный globals
        safe_globals = {
            '__builtins__': {name: getattr(__builtins__, name) 
                            for name in self.safe_builtins 
                            if hasattr(__builtins__, name)}
        }
        
        # Перенаправляем stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Выполняем код с ограниченными глобальными переменными
            exec(code, safe_globals)
            
            result['success'] = True
            result['output'] = captured_output.getvalue()
            result['variables'] = {
                k: v for k, v in safe_globals.items() 
                if not k.startswith('__')
            }
        except Exception as e:
            result['error'] = traceback.format_exc()
            logger.error(f"Code execution failed: {e}")
        finally:
            sys.stdout = old_stdout
        
        return result
    
    def test_feature(self, feature_code: str, test_code: str) -> Dict:
        """Тестирование нового функционала"""
        # Сначала проверяем сам код фичи
        feature_test = self.test_code(feature_code)
        if not feature_test['success']:
            return {
                'feature_valid': False,
                'test_passed': False,
                'feature_error': feature_test['error'],
                'test_error': '',
                'test_output': ''
            }
        
        # Затем тестируем тестовый код
        test_result = self.test_code(test_code)
        if not test_result['success']:
            return {
                'feature_valid': True,
                'test_passed': False,
                'feature_error': '',
                'test_error': test_result['error'],
                'test_output': test_result['output']
            }
        
        # Если оба выполнены успешно
        return {
            'feature_valid': True,
            'test_passed': True,
            'feature_error': '',
            'test_error': '',
            'test_output': test_result['output']
        }
