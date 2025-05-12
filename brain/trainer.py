import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from brain.evaluator import ModelEvaluator
from typing import Optional, Dict, List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JsonDataset(Dataset):
    """Dataset для загрузки и обработки данных диалогов из JSON файлов"""
    def __init__(self, file_path: Union[str, Path], tokenizer, max_length: int = 128):
        """
        Args:
            file_path: Путь к JSON файлу с диалогами
            tokenizer: Токенизатор для обработки текста
            max_length: Максимальная длина последовательности
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Обработка каждого диалога в файле
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

    def _add_example(self, text: str) -> None:
        """Токенизирует и добавляет текст в датасет"""
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.examples.append(encodings)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.examples[idx]['input_ids'].squeeze(),
            'attention_mask': self.examples[idx]['attention_mask'].squeeze()
        }

class SinTrainer:
    """Класс для обучения модели Sin с поддержкой различных датасетов и мониторинга"""
    def __init__(self, model):
        """
        Args:
            model: Экземпляр модели Sin для обучения
        """
        self.model = model
        self.device = model.device
        self.monitor = None
        self.logger = logging.getLogger(__name__)
        self.tokenizer = model.tokenizer

    def evaluate(self, dataset: Dataset, sample_size: int = 100) -> Dict[str, float]:
        """
        Оценка модели на датасете
        
        Args:
            dataset: Датасет для оценки
            sample_size: Количество примеров для оценки
            
        Returns:
            Словарь с метриками (loss, accuracy, perplexity, similarity)
        """
        evaluator = ModelEvaluator(self.model, self.model.tokenizer)
        metrics = evaluator.evaluate_dataset(dataset, sample_size)
        
        # Дополнительный расчет loss
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

    def get_data_loader(self, dataset: Dataset, batch_size: int = 4) -> DataLoader:
        """Создает DataLoader для заданного датасета"""
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Выполняет один шаг обучения на батче данных"""
        inputs = batch['input_ids'].to(self.device)
        masks = batch['attention_mask'].to(self.device)
        outputs = self.model(inputs, attention_mask=masks)
        return F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            inputs.view(-1),
            ignore_index=self.model.tokenizer.pad_token_id
        )

    def load_json_data(self, file_path: Union[str, Path]) -> Dataset:
        """Загружает данные диалогов из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._create_dataset(data['dialogues'])

    def load_text_data(self, file_path: Union[str, Path]) -> Dataset:
        """Загружает текстовые данные из файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return self._create_dataset([{'user_query': t, 'responses': ['']} for t in texts])

    def _create_dataset(self, dialogues: List[Dict]) -> Dataset:
        """Создает датасет из списка диалогов"""
        texts = []
        for dialogue in dialogues:
            query = dialogue.get('user_query', '')
            for response in dialogue.get('responses', []):
                text = response.get('text', '') if isinstance(response, dict) else str(response)
                texts.append(f"Пользователь: {query}\nSin: {text}")
        return self.TextDataset(texts, self.model.tokenizer)

    class TextDataset(Dataset):
        """Dataset для работы с текстовыми данными"""
        def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
            self.encodings = tokenizer(
                texts, 
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        def __len__(self) -> int:
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx]
            }

    def train(self, dataset: Dataset, epochs: int = 3, batch_size: int = 4, 
             lr: float = 5e-5) -> Dict[str, Union[str, float, Dict]]:
        """
        Основной метод обучения модели
        
        Args:
            dataset: Датасет для обучения
            epochs: Количество эпох
            batch_size: Размер батча
            lr: Скорость обучения
            
        Returns:
            Словарь с результатами обучения
        """
        try:
            # Проверка входных данных
            if dataset is None or len(dataset) == 0:
                error_msg = "Dataset is empty or None"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if epochs <= 0 or batch_size <= 0:
                error_msg = f"Invalid parameters: epochs={epochs}, batch_size={batch_size}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Инициализация
            self.logger.info(f"Starting training for {epochs} epochs, batch_size={batch_size}, lr={lr}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
            # Начальная оценка
            self.logger.info("Running initial evaluation...")
            initial_metrics = self.evaluate(dataset)
            self.logger.info(f"Initial metrics: {initial_metrics}")

            # Настройка оптимизатора
            optimizer = AdamW(self.model.parameters(), lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=100,
                num_training_steps=len(dataloader) * epochs
            )

            # Цикл обучения
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
                    
                        # Прямой и обратный проход
                        loss = self.train_step(batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                    
                        total_loss += loss.item()
                        processed_batches += 1
                    
                        # Логирование прогресса
                        if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                            avg_loss = total_loss / processed_batches
                            self.logger.debug(
                                f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                                f"| Loss: {avg_loss:.4f}"
                            )

                    # Логирование после эпохи
                    avg_epoch_loss = total_loss / len(dataloader)
                    self.logger.info(f"Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
                
                    # Валидация после эпохи
                    epoch_metrics = self.evaluate(dataset)
                    self.logger.info(f"Epoch {epoch+1} metrics: {epoch_metrics}")
                
                    if self.monitor is not None:
                        self.monitor.log_epoch(epoch+1, avg_epoch_loss, epoch_metrics)
                    
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        self.logger.error("CUDA out of memory - try reducing batch size")
                        return {
                            'status': 'error',
                            'message': 'CUDA out of memory',
                            'suggestion': 'Try reducing batch size'
                        }
                    raise

            # Финальная оценка
            self.model.eval()
            self.logger.info("Training complete. Running final evaluation...")
            final_metrics = self.evaluate(dataset)
        
            # Формирование отчета
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
            return report
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'exception_type': type(e).__name__
            }
