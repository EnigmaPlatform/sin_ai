# Система подтверждения
import ast
import re
import hashlib
from typing import Dict, List
import logging

class EnhancedSafetyChecker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_patterns = [
            (r"os\.system\(", "system_call"),
            (r"subprocess\.", "subprocess_call"),
            (r"open\(.*, ['\"]w['\"]\)", "file_write"),
            (r"eval\(", "eval_call"),
            (r"__import__\(", "dynamic_import"),
            (r"requests\.(get|post|put|delete)\(", "network_call"),
            (r"socket\.", "socket_operation"),
            (r"pickle\.", "pickle_operation")
        ]
        self.allowed_hashes = {
            "standard_functions": set()
        }

    def analyze_code(self, code: str) -> Dict:
        """Полный анализ кода на безопасность"""
        self.logger.info("Starting code safety analysis")
        
        # Базовые проверки
        syntax_ok, syntax_error = self._check_syntax(code)
        if not syntax_ok:
            return {
                "safe": False,
                "risks": [f"Syntax error: {syntax_error}"],
                "permissions_required": []
            }

        # Проверка шаблонов рисков
        risks = self._detect_risk_patterns(code)
        
        # Проверка импортов
        imports = self._analyze_imports(code)
        risks.extend(imports.get("risky_imports", []))
        
        # Проверка хэшей (для доверенных фрагментов)
        is_trusted = self._check_trusted_fragments(code)
        
        return {
            "safe": len(risks) == 0 and is_trusted,
            "risks": risks,
            "permissions_required": list(set([risk[1] for risk in risks])),
            "import_analysis": imports,
            "trusted": is_trusted
        }

    def _check_syntax(self, code: str) -> tuple:
        """Проверка синтаксиса Python"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def _detect_risk_patterns(self, code: str) -> List[tuple]:
        """Обнаружение опасных паттернов"""
        detected = []
        for pattern, risk_type in self.risk_patterns:
            if re.search(pattern, code):
                detected.append((f"Found {risk_type} pattern", risk_type))
        return detected

    def _analyze_imports(self, code: str) -> Dict:
        """Анализ импортов в коде"""
        try:
            tree = ast.parse(code)
            imports = set()
            risky_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module)
            
            return {
                "imports": list(imports),
                "risky_imports": risky_imports
            }
        except:
            return {"imports": [], "risky_imports": []}

    def _check_trusted_fragments(self, code: str) -> bool:
        """Проверка хэшей доверенных фрагментов кода"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        return code_hash in self.allowed_hashes["standard_functions"]

    def add_trusted_fragment(self, code: str) -> None:
        """Добавление доверенного фрагмента кода"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.allowed_hashes["standard_functions"].add(code_hash)
        self.logger.info(f"Added trusted fragment with hash: {code_hash[:8]}...")
