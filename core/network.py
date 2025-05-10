import torch
import logging
from .file_manager import FileManager
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

class SinNetwork:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained("ai-forever/rugpt3small_based_on_gpt2").to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
        self.context = []
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def learn_from_file(self, file_path: str):
        """Обучение из файла с использованием FileManager"""
        try:
            text = FileManager.extract_text_from_file(file_path)
            self.learn_from_text(text)
            logger.info(f"Успешно обучено из файла: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка обучения из файла: {str(e)}")
            raise
        
    def communicate(self, message: str) -> str:
        try:
            self.context.append(f"User: {message}")
            input_text = "\n".join(self.context[-3:])
            
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.context.append(f"AI: {response}")
            return response
        except Exception as e:
            logger.error(f"Communication error: {str(e)}")
            return "Произошла ошибка при генерации ответа"
        
    def learn_from_text(self, text: str):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        except Exception as e:
            logger.error(f"Learning error: {str(e)}")
