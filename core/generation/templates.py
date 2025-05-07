import jinja2
from pathlib import Path
from typing import Dict, Any
import logging
import json

class TemplateEngine:
    def __init__(self, templates_dir="templates"):
        self.templates_dir = Path(templates_dir)
        self.logger = logging.getLogger("TemplateEngine")
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self._load_template_metadata()

    def _load_template_metadata(self):
        """Загрузка метаданных шаблонов"""
        self.metadata = {}
        meta_file = self.templates_dir / "templates.json"
        
        if meta_file.exists():
            with open(meta_file) as f:
                self.metadata = json.load(f)
        else:
            self.logger.warning(f"No template metadata found at {meta_file}")

    def generate_from_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Генерация кода/текста из шаблона"""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except jinja2.TemplateNotFound:
            self.logger.error(f"Template not found: {template_name}")
            raise
        except Exception as e:
            self.logger.error(f"Template rendering failed: {str(e)}")
            raise

    def list_templates(self) -> Dict[str, Dict]:
        """Список доступных шаблонов с описанием"""
        return {
            t.stem: {
                "description": self.metadata.get(t.stem, {}).get("description", ""),
                "parameters": self.metadata.get(t.stem, {}).get("parameters", [])
            }
            for t in self.templates_dir.glob("*.jinja2")
        }

    def validate_context(self, template_name: str, context: Dict[str, Any]) -> bool:
        """Проверка соответствия контекста требованиям шаблона"""
        if template_name not in self.metadata:
            return True
            
        required_params = set(self.metadata[template_name].get("required_params", []))
        provided_params = set(context.keys())
        
        return required_params.issubset(provided_params)

    def create_template(self, name: str, content: str, metadata: Dict = None):
        """Создание нового шаблона"""
        template_file = self.templates_dir / f"{name}.jinja2"
        template_file.write_text(content)
        
        if metadata:
            self.metadata[name] = metadata
            self._save_metadata()
            
        self.logger.info(f"Created new template: {name}")

    def _save_metadata(self):
        """Сохранение метаданных шаблонов"""
        meta_file = self.templates_dir / "templates.json"
        with open(meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
