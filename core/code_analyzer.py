# core/code_analyzer.py
import ast
import logging
from typing import Dict, List
import hashlib

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java']
    
    def analyze(self, code: str, language: str = 'python') -> Dict:
        """Анализ кода и извлечение информации."""
        if language not in self.supported_languages:
            logger.warning(f"Language {language} is not fully supported")
        
        analysis = {
            'code': code,
            'language': language,
            'hash': self._generate_hash(code),
            'analysis': '',
            'structure': {},
            'complexity': 0,
            'dependencies': []
        }
        
        if language == 'python':
            analysis.update(self._analyze_python(code))
        
        return analysis
    
    def _analyze_python(self, code: str) -> Dict:
        """Анализ Python кода."""
        try:
            tree = ast.parse(code)
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'structure': self._extract_structure(tree),
                'complexity': self._calculate_complexity(tree)
            }
            
            # Извлечение функций, классов и импортов
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'lineno': node.lineno
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'lineno': node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module if node.module else ''
                    analysis['imports'].extend(
                        f"{module}.{alias.name}" for alias in node.names
                    )
            
            return analysis
        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
            return {
                'error': str(e),
                'functions': [],
                'classes': [],
                'imports': []
            }
    
    def _extract_structure(self, tree: ast.AST) -> Dict:
        """Извлечение структуры кода."""
        # Упрощенная реализация
        return {'node_count': len(list(ast.walk(tree)))}
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Вычисление сложности кода."""
        # Упрощенная метрика сложности
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 2
            elif isinstance(node, ast.ClassDef):
                complexity += 3
        return complexity
    
    def _generate_hash(self, code: str) -> str:
        """Генерация хеша кода для идентификации."""
        return hashlib.md5(code.encode('utf-8')).hexdigest()
