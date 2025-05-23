import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from typing import Optional, Dict, List, Union, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DialogDataset(Dataset):
    def __init__(self, data: Union[List[Dict], str, Path], tokenizer, max_length: int = 128, format: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        if isinstance(data, (str, Path)):
            self._load_from_file(data, format)
        else:
            self._process_data(data)
    
    def _load_from_file(self, file_path: Union[str, Path], format: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("File is empty")
                
                if format == "json":
                    data = json.loads(content)
                    if isinstance(data, dict) and 'dialogues' in data:
                        data = data['dialogues']
                    elif not isinstance(data, list):
                        data = [data]
                elif format == "text":
                    data = [{"user_query": line.strip(), "responses": [""]} 
                           for line in content.split('\n') if line.strip()]
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                self._process_data(data)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def _process_data(self, data: List[Dict]):
        for dialog in data:
            if not isinstance(dialog, dict):
                continue
                
            query = dialog.get('user_query', '').strip()
            if not query:
                continue
                
            responses = dialog.get('responses', [])
            if not responses:
                self._add_example(query, "")
                continue
                
            for response in responses:
                if isinstance(response, dict):
                    answer = response.get('text', '').strip()
                else:
                    answer = str(response).strip()
                
                self._add_example(query, answer)

    def _add_example(self, query: str, answer: str):
        """Генерирует правильные labels для задачи языкового моделирования"""
        text = f"User: {query}\nAssistant: {answer}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Создаем labels, смещенные на 1 токен вперед
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Сдвигаем на один токен вперед
        labels[-1] = -100  # Игнорируем последний токен
        
        self.examples.append({
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class SinTrainer:
    def __init__(self, model, device: str = None):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

    def get_data_loader(self, dataset, batch_size=4, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }

    def train_step(self, batch):
        """Обновленный метод с обработкой возвращаемого значения"""
        inputs = batch['input_ids'].to(self.device)
        masks = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(inputs, attention_mask=masks, labels=labels)
        return outputs['loss']

    def load_dataset(self, file_path: Union[str, Path]) -> Dataset:
        """
        Загружает датасет из файла (для обратной совместимости)
        
        Args:
            file_path: Путь к файлу с данными
            
        Returns:
            Dataset: Загруженный датасет
        """
        return DialogDataset(file_path, self.tokenizer, format=file_path.suffix[1:])
        
    def create_dataloader(self, 
                         data: Union[List[Dict], str, Path],
                         batch_size: int = 4,
                         shuffle: bool = True,
                         **kwargs) -> DataLoader:
        """
        Создает DataLoader для данных
        
        Args:
            data: Входные данные (путь или список диалогов)
            batch_size: Размер батча
            shuffle: Перемешивать ли данные
            kwargs: Доп. параметры для Dataset
            
        Returns:
            DataLoader для переданных данных
        """
        if isinstance(data, (str, Path)):
            dataset = DialogDataset(data, self.tokenizer, format=data.suffix[1:], **kwargs)
        else:
            dataset = DialogDataset(data, self.tokenizer, **kwargs)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Оценка модели на датасете"""
        self.model.eval()
        dataloader = self.get_data_loader(dataset, shuffle=False)
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                total_loss += outputs['loss'].item()
        
        return {'loss': total_loss / len(dataloader)}

    def train(self, 
              train_data: Union[List[Dict], str, Path],
              val_data: Union[List[Dict], str, Path] = None,
              epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              max_grad_norm: float = 1.0,
              logging_steps: int = 10,
              **dataset_kwargs) -> Dict:
        """
        Полный цикл обучения модели
        
        Args:
            train_data: Данные для обучения
            val_data: Данные для валидации (опционально)
            epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Скорость обучения
            warmup_steps: Шаги для разогрева
            max_grad_norm: Макс. норма градиента
            logging_steps: Частота логирования
            dataset_kwargs: Доп. параметры для Dataset
            
        Returns:
            Словарь с результатами обучения
        """
        try:
            # Подготовка данных
            train_loader = self.create_dataloader(
                train_data, 
                batch_size=batch_size,
                **dataset_kwargs
            )
            
            val_loader = None
            if val_data:
                val_loader = self.create_dataloader(
                    val_data,
                    batch_size=batch_size,
                    shuffle=False,
                    **dataset_kwargs
                )

            # Настройка оптимизации
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            # Начальная оценка
            self.logger.info("Running initial evaluation...")
            results = {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'learning_rates': []
            }
            
            # Цикл обучения
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                self.logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
                for step, batch in enumerate(progress_bar):
                    loss = self.train_step(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    if (step + 1) % logging_steps == 0:
                        progress_bar.set_postfix({'loss': loss.item()})
                
                # Сохранение метрик эпохи
                avg_train_loss = epoch_loss / len(train_loader)
                results['epochs'].append(epoch + 1)
                results['train_loss'].append(avg_train_loss)
                results['learning_rates'].append(scheduler.get_last_lr()[0])
                
                self.logger.info(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")
                
                # Валидация
                if val_loader:
                    val_metrics = self.evaluate(val_loader.dataset)
                    results['val_loss'].append(val_metrics['loss'])
                    self.logger.info(f"Epoch {epoch + 1} Val Loss: {val_metrics['loss']:.4f}")
            
            return {
                'status': 'success',
                'results': results,
                'best_epoch': results['val_loss'].index(min(results['val_loss'])) + 1 if val_data else 0
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'exception_type': type(e).__name__
            }

    def save_model(self, path: Union[str, Path]):
        """Сохраняет модель и токенизатор"""
        torch.save({
            'model_state': self.model.state_dict(),
            'tokenizer_config': self.tokenizer.get_vocab()
        }, path)
        self.logger.info(f"Model saved to {path}")
