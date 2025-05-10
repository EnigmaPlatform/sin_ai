import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as F

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
        self.monitor = TrainingMonitor()

    def evaluate(self, dataset, sample_size=100):
        """Оценка качества модели"""
        self.model.eval()
        total_loss = 0
        correct = 0
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i > sample_size // 4:
                    break
                
                inputs = batch["input_ids"].to(self.device)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                                      inputs.view(-1),
                                      ignore_index=self.model.tokenizer.pad_token_id)
                total_loss += loss.item()
        
        return {"loss": total_loss / (sample_size // 4)}

    def load_json_data(self, file_path):
        return JsonDataset(file_path, self.model.tokenizer)

    def load_text_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return self.ConversationDataset(texts, self.model.tokenizer)

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

    def train(self, dataset, epochs=3, batch_size=4, lr=5e-5):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        initial_metrics = self.evaluate(dataset)
        print(f"Initial metrics: {initial_metrics}")
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * epochs
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                
                outputs = self.model(inputs, attention_mask=masks)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1),
                    ignore_index=self.model.tokenizer.pad_token_id
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
        self.model.eval()
        return total_loss / len(dataloader)

# Логирование после каждой эпохи
            epoch_metrics = self.evaluate(dataset)
            self.monitor.log_epoch(epoch+1, total_loss/len(dataloader), epoch_metrics)
        
        # Оценка после обучения
        final_metrics = self.evaluate(dataset)
        print(f"\nTraining complete!")
        print(f"Initial loss: {initial_metrics['loss']:.4f}")
        print(f"Final loss: {final_metrics['loss']:.4f}")
        print(f"Improvement: {initial_metrics['loss'] - final_metrics['loss']:.4f}")
        
        return final_metrics
