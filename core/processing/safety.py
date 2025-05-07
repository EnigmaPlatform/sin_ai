# Система подтверждения
import ast
import re

class CodeSafetyChecker:
    RISK_PATTERNS = [
        r"os\.system\(",
        r"subprocess\.run\(",
        r"open\(.*, ['\"]w['\"]\)",
        r"eval\(",
        r"__import__\("
    ]

    def analyze(self, code):
        risks = []
        for pattern in self.RISK_PATTERNS:
            if re.search(pattern, code):
                risks.append(f"Обнаружен риск: {pattern}")
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            risks.append(f"Синтаксическая ошибка: {str(e)}")
        
        return {
            'is_safe': len(risks) == 0,
            'risks': risks,
            'required_permissions': self._detect_permissions(code)
        }

    def _detect_permissions(self, code):
        perms = []
        if "open(" in code:
            perms.append("file_io")
        if "requests.get" in code:
            perms.append("network")
        return perms
