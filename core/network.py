import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

class SinNetwork:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained("ai-forever/rugpt3small_based_on_gpt2").to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
        self.context = []
        
    def communicate(self, message: str) -> str:
        self.context.append(f"User: {message}")
        input_text = "\n".join(self.context[-3:])
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.context.append(f"AI: {response}")
        return response
        
    def learn_from_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
