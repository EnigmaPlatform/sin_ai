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

def generate_response(self, prompt, max_new_tokens=100, **kwargs):
    inputs = self.tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(self.device)
    
    with torch.no_grad():
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
    
    return self.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    )

def save(self, path):
    # Убедитесь, что сохраняете только state_dict
    torch.save({
        'model_state': self.state_dict(),
        'tokenizer_config': self.tokenizer.get_vocab()
    }, path)

@classmethod
def load(cls, path):
    model = cls()
    try:
        # Сначала пробуем безопасную загрузку
        state = torch.load(path, map_location=model.device, weights_only=True)
    except:
        try:
            # Затем пробуем legacy-загрузку
            state = torch.load(path, map_location=model.device, weights_only=False)
        except Exception as e:
            # Если всё равно ошибка - файл повреждён
            print(f"Файл модели повреждён. Удалите {path} и перезапустите программу")
            raise e
    
    model.load_state_dict(state['model_state'])
    return model
