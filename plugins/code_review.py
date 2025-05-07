# Анализ качества кода
import ast

class CodeReviewPlugin:
    def analyze(self, code):
        issues = []
        tree = ast.parse(code)
        
        # Поиск длинных функций
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 20:
                    issues.append(f"Функция {node.name} слишком длинная")
        
        # Проверка стиля
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and len(node.id) < 3:
                issues.append(f"Слишком короткое имя: {node.id}")
        
        return issues
