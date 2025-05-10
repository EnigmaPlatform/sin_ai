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
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self._create_dataset(data['dialogues'])

    def load_text_data(self, file_path):
        with open(file_path, 'r') as f:
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

    class TextDataset(torch.utils.data.Dataset):
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
