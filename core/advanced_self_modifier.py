# core/advanced_self_modifier.py
import ast
import difflib
import hashlib
import subprocess
from pathlib import Path

class AdvancedSelfModifier:
    def __init__(self, model_root: str):
        self.model_root = Path(model_root)
        self.backup_dir = self.model_root / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.MAX_SIZE = 64 * 1024 * 1024  # 64MB

    def propose_change(self, new_code: str, change_desc: str, test_script: str = None) -> dict:
        """Полный цикл предложения изменений"""
        # Проверка размера
        if len(new_code.encode('utf-8')) > self.MAX_SIZE:
            return {"status": "error", "reason": "size_limit_exceeded"}
        
        # Анализ безопасности
        safety_report = self._check_code_safety(new_code)
        if not safety_report["safe"]:
            return {"status": "rejected", "reason": "safety_risk", "details": safety_report}
        
        # Тестирование в песочнице
        test_result = self._run_in_sandbox(new_code, test_script)
        if not test_result["passed"]:
            return {"status": "rejected", "reason": "test_failed", "details": test_result}
        
        # Генерация diff
        current_code = self._get_current_code()
        diff = self._generate_diff(current_code, new_code)
        
        return {
            "status": "requires_approval",
            "diff": diff,
            "performance_impact": self._estimate_performance_impact(new_code),
            "memory_impact": self._estimate_memory_impact(new_code),
            "test_results": test_result,
            "description": change_desc
        }

    def apply_change(self, new_code: str) -> bool:
        """Финализация изменений"""
        try:
            # Создание резервной копии
            backup_path = self.backup_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            with open(backup_path, 'w') as f:
                f.write(self._get_current_code())
            
            # Применение изменений
            with open(self.model_root / "core" / "model.py", 'w') as f:
                f.write(new_code)
            
            # Перезагрузка модели
            self._reload_model()
            return True
        except Exception as e:
            self._restore_backup(backup_path)
            return False

    def _run_in_sandbox(self, code: str, test_script: str = None) -> dict:
        """Запуск в изолированном окружении с тестами"""
        sandbox_env = {
            '__builtins__': None,
            'print': print,
            'math': __import__('math'),
            'numpy': __import__('numpy')
        }
        
        try:
            # Исполнение основного кода
            exec(code, sandbox_env)
            
            # Запуск тестов если есть
            if test_script:
                exec(test_script, sandbox_env)
            
            return {"passed": True}
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _check_code_safety(self, code: str) -> dict:
        """Расширенный анализ безопасности"""
        risk_patterns = {
            "file_operations": r"(open|os\.remove|shutil\.)",
            "network": r"(requests\.|socket\.)",
            "system": r"(subprocess\.|os\.system)"
        }
        
        risks = []
        for category, pattern in risk_patterns.items():
            if re.search(pattern, code):
                risks.append(category)
        
        return {
            "safe": len(risks) == 0,
            "detected_risks": risks,
            "required_permissions": risks
        }
