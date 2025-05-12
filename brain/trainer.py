import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from brain.evaluator import ModelEvaluator
from typing import Optional, Dict, List, Union, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DialogDataset(Dataset):
    def __init__(self, 
                 data: Union[List[Dict], str, Path], 
                 tokenizer, 
                 max_length: int = 128,
                 format: str = None):
        """
        Args:
            data: Может быть путем к файлу или готовым списком диалогов
            tokenizer: Токенизатор для обработки текста
            max_length: Максимальная длина последовательности
            format: Формат данных ('json' или 'text'), если data - путь
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Если data - это путь, загружаем из файла
        if isinstance(data, (str, Path)):
            self._load_from_file(data, format)
        else:
            # Иначе обрабатываем как готовые данные
            self._process_data(data)
    
    def _load_from_file(self, file_path: Union[str, Path], format: str):
        """Загружает данные из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("File is empty")
                
                if format == "json":
                    data = json.loads(content)
                    # Обрабатываем структуру с полем "dialogues"
                    if isinstance(data, dict) and 'dialogues' in data:
                        data = data['dialogues']
                    elif not isinstance(data, list):
                        data = [data]  # Преобразуем в список, если это не список
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
        """Обрабатывает данные диалогов"""
        for dialog in data:
            if not isinstance(dialog, dict):
                continue
                
            # Получаем запрос пользователя
            query = dialog.get('user_query', '').strip()
            if not query:
                continue
                
            # Обрабатываем ответы
            responses = dialog.get('responses', [])
            if not responses:
                # Если нет ответов, добавляем пустой ответ
                self._add_example(query, "")
                continue
                
            for response in responses:
                if isinstance(response, dict):
                    answer = response.get('text', '').strip()
                else:
                    answer = str(response).strip()
                
                self._add_example(query, answer)

    def _add_example(self, query: str, answer: str):
        """Добавляет пример диалога в датасет"""
        text = f"User: {query}\nAssistant: {answer}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.examples.append({
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)  # Добавляем labels для обучения
        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class SinTrainer:
    """Улучшенный класс для обучения модели с дополнительными функциями"""
    def __init__(self, model, device: str = None):
        """
        Args:
            model: Экземпляр модели для обучения
            device: Устройство для вычислений (auto, cuda, cpu)
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

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

    def _collate_fn(self, batch):
        """Функция для объединения примеров в батчи"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])  # Добавляем labels
    }

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
                    # Перенос данных на устройство
                    inputs = batch['input_ids'].to(self.device)
                    masks = batch['attention_mask'].to(self.device)
                    
                    # Прямой проход
                    outputs = self.model(inputs, attention_mask=masks)
                    loss = F.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        inputs.view(-1),
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    
                    # Обратный проход
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Логирование
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
                    val_loss = self.evaluate(val_loader)
                    results['val_loss'].append(val_loss)
                    self.logger.info(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f}")
            
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

    def evaluate(self, dataloader: DataLoader) -> float:
        """Оценка модели на данных из DataLoader"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                
                outputs = self.model(inputs, attention_mask=masks)
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    inputs.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def save_model(self, path: Union[str, Path]):
        """Сохраняет модель и токенизатор"""
        torch.save({
            'model_state': self.model.state_dict(),
            'tokenizer_config': self.tokenizer.get_vocab()
        }, path)
        self.logger.info(f"Model saved to {path}")

   def get_data_loader(self, dataset, batch_size=4, shuffle=True):
       """
    Создает DataLoader для переданного датасета
    
    Args:
        dataset: Загруженный датасет
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        
    Returns:
        DataLoader для переданного датасета
    """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
    )
