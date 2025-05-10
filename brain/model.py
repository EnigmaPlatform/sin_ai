import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class SinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Sin"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
        self.base_model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2").to(self.device)

        self.adaptation = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        ).to(self.device)

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.base_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        adapted = self.adaptation(outputs.hidden_states[-1])
        return self.base_model.lm_head(adapted)

    def generate_response(self, prompt, max_length=100, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path):
        """Безопасное сохранение только необходимых данных"""
        torch.save({
            'model_state': self.state_dict(),
            'tokenizer_config': {
                'vocab': self.tokenizer.get_vocab(),
                'special_tokens': self.tokenizer.special_tokens_map
            }
        }, path)

    @classmethod 
    def load(cls, path):
        """Безопасная загрузка"""
        model = cls()
        try:
        # Сначала пробуем безопасный режим
            state = torch.load(path, map_location=model.device, weights_only=True)
        except:
        # Если не получается, пробуем небезопасный (только для доверенных файлов!)
            state = torch.load(path, map_location=model.device, weights_only=False)
    
        model.load_state_dict(state['model_state'])
    
    # Восстанавливаем токенизатор
        if 'tokenizer_config' in state:
            model.tokenizer.add_special_tokens(state['tokenizer_config']['special_tokens'])
    
        return model
