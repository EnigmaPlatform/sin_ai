import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging

logger = logging.getLogger(__name__)

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
        self.base_model = self.base_model.to(torch.bfloat16)  # 16-битная точность
        self.base_model.eval()  # Включаем режим оценки для части оптимизаций
    
    # Добавьте после инициализации модели:
    torch._C._jit_set_texpr_fuser_enabled(False)  # Отключаем JIT для стабильности
        
        # 3. Настройка параметров
        self.base_model.config.pad_token_id = self.tokenizer.eos_token_id
        self.base_model.config.max_length = 512
        
        # 4. Адаптационные слои
        self.adaptation = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        ).to(self.device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Основной forward pass модели
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # Адаптация скрытых состояний
        adapted = self.adaptation(outputs.hidden_states[-1])
        logits = self.base_model.lm_head(adapted)
        
        # Вычисление loss если есть labels
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}

    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7):
        try:
            # Очистка промпта от предыдущих ответов
            clean_prompt = prompt.split("Sin:")[0].strip()
            
            inputs = self.tokenizer(
                clean_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Извлекаем только новый сгенерированный текст
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.replace(clean_prompt, "").strip()
            return response.split("\n")[0]  # Берем первую строку ответа
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {str(e)}")
            return "Извините, произошла ошибка"

    def save(self, path):
        """Сохранение модели с обработкой ошибок"""
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
            logger.error(f"Ошибка сохранения: {str(e)}")
            return False

    @classmethod
    def load(cls, path):
        """Безопасная загрузка модели"""
        model = cls()
        try:
            state = torch.load(path, map_location=model.device, weights_only=False)
            model.load_state_dict(state['model_state'])
            if 'tokenizer_config' in state:
                model.tokenizer.add_special_tokens(state['tokenizer_config']['special_tokens'])
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}\nУдалите файл модели и создайте новую")
            raise
