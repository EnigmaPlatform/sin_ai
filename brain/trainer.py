import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from brain.evaluator import ModelEvaluator
import logging
logger = logging.getLogger(__name__)

class JsonDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for dialogue in data.get('dialogues', []):
            user_query = dialogue.get('user_query', '')
            for response in dialogue.get('responses', []):
                if isinstance(response, dict):
                    answer = response.get('text', '')
                else:
                    answer = str(response)
                
                if user_query and answer:
                    text = f"Пользователь: {user_query}\nSin: {answer}"
                    self._add_example(text)

    def _add_example(self, text):
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.examples.append(encodings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'].squeeze(),
            'attention_mask': self.examples[idx]['attention_mask'].squeeze()
        }

class SinTrainer:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.monitor = None  # Добавим атрибут monitor для совместимости

    def evaluate(self, dataset, sample_size=100):
        """Оценка модели на датасете с вычислением различных метрик"""
        # Инициализация ModelEvaluator
        evaluator = ModelEvaluator(self.model, self.model.tokenizer)
        
        # Вычисление метрик на датасете
        metrics = evaluator.evaluate_dataset(dataset, sample_size)
        
        # Дополнительные вычисления для loss
        dataloader = DataLoader(dataset, batch_size=4)
        total_loss = 0
        count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                
                outputs = self.model(inputs, attention_mask=masks)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1),
                    ignore_index=self.model.tokenizer.pad_token_id
                )
                
                total_loss += loss.item()
                count += 1
        
        metrics['loss'] = total_loss / count if count > 0 else 0.0
        
        return {
            'loss': metrics['loss'],
            'accuracy': metrics.get('accuracy', 0.0),
            'perplexity': metrics.get('perplexity', 0.0),
            'similarity': metrics.get('semantic_similarity', 0.0)
        }

    def get_data_loader(self, dataset, batch_size=4):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_step(self, batch):
        inputs = batch['input_ids'].to(self.device)
        masks = batch['attention_mask'].to(self.device)
        outputs = self.model(inputs, attention_mask=masks)
        return torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            inputs.view(-1),
            ignore_index=self.model.tokenizer.pad_token_id
        )

    def load_json_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._create_dataset(data['dialogues'])

    def load_text_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return self._create_dataset([{'user_query': t, 'responses': ['']} for t in texts])

    def _create_dataset(self, dialogues):
        texts = []
        for dialogue in dialogues:
            query = dialogue.get('user_query', '')
            for response in dialogue.get('responses', []):
                text = response.get('text', '') if isinstance(response, dict) else str(response)
                texts.append(f"Пользователь: {query}\nSin: {text}")
        return self.TextDataset(texts, self.model.tokenizer)

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.encodings = tokenizer(
                texts, 
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx]
            }

    class ConversationDataset(Dataset):
        def __init__(self, texts, tokenizer, block_size=128):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt"
            )
            
        def __len__(self):
            return len(self.encodings["input_ids"])
            
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx]
            }

    def evaluate(self, dataset, sample_size=100):
        """Оценка модели на датасете с вычислением различных метрик"""
    # Инициализация ModelEvaluator
        evaluator = ModelEvaluator(self.model, self.model.tokenizer)
    
    # Вычисление метрик на датасете
        metrics = evaluator.evaluate_dataset(dataset, sample_size)
    
    # Дополнительные вычисления для loss (если нужно)
        dataloader = DataLoader(dataset, batch_size=4)
        total_loss = 0
        count = 0
    
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
            
                outputs = self.model(inputs, attention_mask=masks)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1),
                    ignore_index=self.model.tokenizer.pad_token_id
            )
            
                total_loss += loss.item()
                count += 1
    
        metrics['loss'] = total_loss / count if count > 0 else 0.0
    
        return {
            'loss': metrics['loss'],
            'accuracy': metrics.get('accuracy', 0.0),
            'perplexity': metrics.get('perplexity', 0.0),
            'similarity': metrics.get('semantic_similarity', 0.0)
    }

    def train(self, dataset, epochs=3, batch_size=4, lr=5e-5):
        """Улучшенный метод обучения с проверками и логированием"""
        try:
        # 1. Проверка входных данных
            if dataset is None or len(dataset) == 0:
                error_msg = "Dataset is empty or None"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if epochs <= 0 or batch_size <= 0:
                error_msg = f"Invalid parameters: epochs={epochs}, batch_size={batch_size}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # 2. Инициализация
            self.logger.info(f"Starting training for {epochs} epochs, batch_size={batch_size}, lr={lr}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 3. Начальная оценка
            self.logger.info("Running initial evaluation...")
            initial_metrics = self.evaluate(dataset)
            self.logger.info(f"Initial metrics: {initial_metrics}")
            print(f"\nInitial metrics: {initial_metrics}")

        # 4. Настройка оптимизатора
            optimizer = AdamW(self.model.parameters(), lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=100,
                num_training_steps=len(dataloader) * epochs
        )

        # 5. Цикл обучения
            self.model.train()
            for epoch in range(epochs):
                self.logger.info(f"Starting epoch {epoch+1}/{epochs}")
                total_loss = 0
                processed_batches = 0
            
                try:
                    for batch_idx, batch in enumerate(dataloader):
                    # Проверка наличия необходимых ключей
                        if 'input_ids' not in batch or 'attention_mask' not in batch:
                            self.logger.warning(f"Skipping invalid batch at index {batch_idx}")
                            continue

                        optimizer.zero_grad()
                    
                    # Перенос данных на устройство
                        inputs = batch["input_ids"].to(self.device)
                        masks = batch["attention_mask"].to(self.device)
                    
                    # Прямой проход
                        outputs = self.model(inputs, attention_mask=masks)
                    
                    # Расчет потерь
                        loss = F.cross_entropy(
                            outputs.view(-1, outputs.size(-1)),
                            inputs.view(-1),
                            ignore_index=self.model.tokenizer.pad_token_id
                    )
                    
                    # Обратный проход
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                    
                        total_loss += loss.item()
                        processed_batches += 1
                    
                    # Логирование каждые 10% батчей
                        if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                            avg_loss = total_loss / processed_batches
                            self.logger.debug(
                                f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                                f"| Loss: {avg_loss:.4f}"
                        )

                # 6. Логирование после эпохи
                    avg_epoch_loss = total_loss / len(dataloader)
                    self.logger.info(f"Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f}")
                
                # Валидация после эпохи
                    epoch_metrics = self.evaluate(dataset)
                    self.logger.info(f"Epoch {epoch+1} metrics: {epoch_metrics}")
                
                    if self.monitor is not None:
                        self.monitor.log_epoch(epoch+1, avg_epoch_loss, epoch_metrics)
                    
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        self.logger.error("CUDA out of memory - try reducing batch size")
                        print("\nОшибка: Недостаточно памяти GPU. Попробуйте уменьшить batch_size.")
                        return {
                            'status': 'error',
                            'message': 'CUDA out of memory',
                            'suggestion': 'Try reducing batch size'
                    }
                    raise

        # 7. Финальная оценка
            self.model.eval()
            self.logger.info("Training complete. Running final evaluation...")
            final_metrics = self.evaluate(dataset)
        
        # 8. Формирование отчета
            report = {
                'initial_metrics': initial_metrics,
                'final_metrics': final_metrics,
                'improvement': {
                    'loss': initial_metrics['loss'] - final_metrics['loss'],
                    'accuracy': final_metrics['accuracy'] - initial_metrics['accuracy']
            },
                'epochs_trained': epochs,
                'status': 'success'
        }
        
            self.logger.info(f"Training report: {report}")
            print("\nTraining complete!")
            print(f"Initial loss: {initial_metrics['loss']:.4f}")
            print(f"Final loss: {final_metrics['loss']:.4f}")
            print(f"Improvement: {initial_metrics['loss'] - final_metrics['loss']:.4f}")
        
            return report
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            print(f"\nОшибка обучения: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'exception_type': type(e).__name__
        }
