import shutil
from pathlib import Path

class FileManager:
    def __init__(self, root_dir="sin_ai"):
        self.root = Path(root_dir)

    def write_file(self, path: str, content: str):
        target = self.root / path
        target.parent.mkdir(exist_ok=True)
        target.write_text(content)

    def update_code(self, module_path: str, new_code: str):
        # Автоматическое обновление кода с бэкапом
        backup = self.root / f"{module_path}.bak"
        original = self.root / f"{module_path}.py"
        
        shutil.copy(original, backup)
        self.write_file(module_path, new_code)
        
        # Перезагрузка модуля
        import importlib
        module = importlib.import_module(module_path.replace('/', '.'))
        importlib.reload(module)
