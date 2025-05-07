# Мультимодальная обработка
from PIL import Image
import pytesseract

class MultimodalProcessor:
    def process(self, input_data):
        if isinstance(input_data, str):
            return self._process_text(input_data)
        elif isinstance(input_data, bytes):
            return self._process_image(input_data)
        else:
            raise ValueError("Unsupported input type")

    def _process_text(self, text):
        # Анализ эмоциональной окраски
        from transformers import pipeline
        classifier = pipeline("text-classification")
        sentiment = classifier(text)[0]
        return {'type': 'text', 'sentiment': sentiment}

    def _process_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return {'type': 'image', 'extracted_text': text}
