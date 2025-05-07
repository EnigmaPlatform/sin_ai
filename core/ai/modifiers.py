import ast
import difflib
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from utils.logger import SinLogger
from utils.config import ConfigManager
from core.utils.safety import CodeSafetyChecker

class CodeModifier:
    def __init__(self, parent_model):
        self.parent = parent_model
        self.logger = SinLogger("CodeModifier")
        self.config = ConfigManager()
        self.safety_checker = CodeSafetyChecker()
        
        # Временное хранилище предложенных изменений
        self.proposed_changes = {}
        self.change_counter = 0
        
        # Настройки безопасности
        self.max_size = self.config.get("safety.max_code_size", 64 * 1024 * 1024)
        self.allowed_imports = self.config.get("safety.allowed_imports", ["math", "numpy"])
    
    def propose_change(self, new_code: str, description: str = "") -> Dict:
        """Предложить изменение кода с проверкой безопасности"""
        # Проверка размера кода
        if len(new_code.encode('utf-8')) > self.max_size:
            return {"status": "rejected", "reason": "code_size_exceeded"}
        
        # Проверка безопасности
        safety_report = self.safety_checker.analyze(new_code)
        if not safety_report["safe"]:
            return {
                "status": "rejected",
                "reason": "safety_risk",
                "details": safety_report
            }
        
        # Тестирование в песочнице
        test_result = self._test_in_sandbox(new_code)
        if not test_result["passed"]:
            return {
                "status": "rejected",
                "reason": "test_failed",
                "details": test_result
            }
        
        # Генерация diff
        current_code = self._get_current_code()
        diff = self._generate_diff(current_code, new_code)
        
        # Сохранение предложения
        change_id = f"change_{self.change_counter}"
        self.proposed_changes[change_id] = {
            "code": new_code,
            "diff": diff,
            "description": description,
            "test_result": test_result,
            "safety_report": safety_report
        }
        self.change_counter += 1
        
        return {
            "status": "requires_approval",
            "change_id": change_id,
            "diff": diff,
            "description": description,
            "test_result": test_result
        }
    
    def apply_change(self, change_id: str) -> bool:
        """Применить предложенное изменение"""
        if change_id not in self.proposed_changes:
            return False
        
        change = self.proposed_changes[change_id]
        
        try:
            # Создание резервной копии
            backup_path = self._create_backup()
            
            # Применение изменений
            with open(self._get_code_path(), "w") as f:
                f.write(change["code"])
            
            # Перезагрузка модели
            self.parent.load(self._get_code_path())
            
            self.logger.info(f"Applied code change: {change_id}")
            return True
        except Exception as e:
            # Восстановление из резервной копии
            self._restore_backup(backup_path)
            self.logger.error(f"Failed to apply change: {str(e)}")
            return False
    
    def _test_in_sandbox(self, code: str) -> Dict:
        """Тестирование кода в изолированной среде"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as temp_file:
            temp_file.write(code)
            temp_file.flush()
            
            try:
                result = subprocess.run(
                    ["python", temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                return {
                    "passed": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
            except subprocess.TimeoutExpired:
                return {"passed": False, "error": "Execution timeout"}
            except Exception as e:
                return {"passed": False, "error": str(e)}
    
    def _get_current_code(self) -> str:
        """Получение текущего кода модели"""
        with open(self._get_code_path()) as f:
            return f.read()
    
    def _get_code_path(self) -> Path:
        """Путь к файлу с кодом модели"""
        return Path(__file__).parent.parent / "model.py"
    
    def _generate_diff(self, old_code: str, new_code: str) -> List[str]:
        """Генерация различий между версиями кода"""
        return list(difflib.unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            lineterm=""
        ))
    
    def _create_backup(self) -> Path:
        """Создание резервной копии кода"""
        backup_dir = Path("data/backups/code")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / f"backup_{len(list(backup_dir.glob('*.bak')))}.bak"
        with open(backup_path, "w") as backup_file:
            backup_file.write(self._get_current_code())
        
        return backup_path
    
    def _restore_backup(self, backup_path: Path) -> None:
        """Восстановление из резервной копии"""
        with open(backup_path) as backup_file:
            with open(self._get_code_path(), "w") as target_file:
                target_file.write(backup_file.read())
