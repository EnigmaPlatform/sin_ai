from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileManager:
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """Чтение текстовых файлов с обработкой ошибок"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")
                
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        """Поддержка основных текстовых форматов"""
        path = Path(file_path)
        if path.suffix == '.txt':
            return FileManager.read_text_file(file_path)
        elif path.suffix == '.pdf':
            from PyPDF2 import PdfReader
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                return "\n".join(page.extract_text() for page in reader.pages)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")
