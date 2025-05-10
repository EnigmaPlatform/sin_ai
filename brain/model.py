import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class SinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Sin"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Инициализация токенизатора
        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # 2. Загрузка основной модели
        self.base_model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2").to(self.device)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # 3. Настройка параметров модели
        self.base_model.config.pad_token_id = self.tokenizer.eos_token_id
        self.base_model.config.max_length = 512
        
        # 4. Дополнительные адаптационные слои
        self.adaptation = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        ).to(self.device)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        adapted = self.adaptation(outputs.hidden_states[-1])
        return self.base_model.lm_head(adapted)

    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """Генерация ответа на основе промпта с обработкой ошибок"""
        try:
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
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Удаляем промпт из ответа
            response = full_response.replace(prompt, "").strip()
            return response.split("\n")[0]  # Берем первую строку ответа
            
        except Exception as e:
            print(f"Ошибка генерации ответа: {str(e)}")
            return "Извините, не могу сформировать ответ"

    def save(self, path):
        """Безопасное сохранение модели"""
        try:
            torch.save({
                'model_state': self.state_dict(),
                'tokenizer_config': {
                    'vocab': self.tokenizer.get_vocab(),
                    'special_tokens': self.tokenizer.special_tokens_map
                }
            }, path)
            return True
        except Exception as e:
            print(f"Ошибка сохранения модели: {str(e)}")
            return False

    @classmethod
    def load(cls, path):
        """Безопасная загрузка модели"""
        model = cls()
        try:
            # Пробуем безопасную загрузку
            state = torch.load(path, map_location=model.device, weights_only=True)
        except:
            try:
                # Пробуем небезопасную загрузку (только для доверенных файлов)
                state = torch.load(path, map_location=model.device, weights_only=False)
            except Exception as e:
                print(f"Файл модели повреждён. Удалите {path} и перезапустите программу")
                raise e
        
        model.load_state_dict(state['model_state'])
        
        # Восстановление токенизатора
        if 'tokenizer_config' in state:
            model.tokenizer.add_special_tokens(state['tokenizer_config']['special_tokens'])
        
        return model
