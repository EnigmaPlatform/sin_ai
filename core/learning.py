import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List, Dict
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
from ..ui.visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class LearningEngine:
    def __init__(self, network):
        self.network = network
        self.learning_rate = 5e-5
        self.batch_size = 4
        self.epochs = 3
        self.learning_speed = 1.0
        self.visualizer = TrainingVisualizer()
    
    def train_on_text(self, text: str) -> None:
        """Обучение на текстовых данных"""
        # Разделение текста на части
        chunks = self._chunk_text(text)
        dataset = TextDataset(chunks, self.network.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Настройка оптимизатора
        optimizer = AdamW(self.network.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * self.epochs
        )
        
        # Обучение
        self.network.train()
        total_steps = len(dataloader) * self.epochs
        progress_bar = tqdm(total=total_steps, desc="Training on text")
        
        for epoch in range(self.epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.network.device)
                attention_mask = batch['attention_mask'].to(self.network.device)
                
                outputs = self.network(input_ids, attention_mask=attention_mask)
                loss = self._calculate_loss(outputs, input_ids)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Обновление прогресса
                progress_bar.update(1)
                self.network.learning_progress = (progress_bar.n / progress_bar.total) * 100
                self.visualizer.update_progress(self.network.learning_progress)
        
        progress_bar.close()
        self.network.eval()
    
    def train_on_code(self, code_analysis: Dict) -> None:
        """Обучение на анализе кода"""
        # Здесь должна быть более сложная логика для обучения на коде
        # В этом примере мы просто используем текст кода как обычный текст
        combined_text = f"Code in {code_analysis['language']}:\n{code_analysis['code']}\n\nAnalysis:\n{code_analysis['analysis']}"
        self.train_on_text(combined_text)
    
    def process_api_response(self, response: Dict) -> None:
        """Обработка ответа от API для обучения"""
        if 'knowledge' in response:
            self.train_on_text(response['knowledge'])
        if 'code' in response:
            self.train_on_text(f"Code example:\n{response['code']}\n\nExplanation:\n{response.get('explanation', '')}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Разделение текста на части"""
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def _calculate_loss(self, outputs, input_ids):
        """Вычисление потерь для языковой модели"""
        # Упрощенная версия - в реальной реализации было бы сложнее
        logits = torch.matmul(outputs, self.network.model.wte.weight.t())
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
