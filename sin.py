import os
import json
import pickle
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import signal
import sys
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import threading
from typing import List, Dict, Tuple, Optional, Any
import traceback
import regex as re

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sin_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SinTokenizer:
    """Кастомный токенизатор для чат-бота Sin с поддержкой DeepSeek словаря"""
    
    def __init__(self, vocab_size: int = 128000):  # Увеличен размер словаря
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = {}
        self.pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        
        # Специальные токены DeepSeek
        self.special_tokens = {
            '<｜begin▁of▁sentence｜>': 0,  # BOS
            '<｜end▁of▁sentence｜>': 1,    # EOS
            '<｜▁pad▁｜>': 2,             # PAD
            '<UNK>': 3                    # UNK
        }
        self._initialize_special_tokens()
        
    def _initialize_special_tokens(self):
        """Инициализация специальных токенов"""
        for token, idx in self.special_tokens.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            
    def load_deepseek_vocab(self, vocab_data: Dict[str, int]) -> bool:
        """Загрузка словаря из данных DeepSeek"""
        try:
            # Сортировка по ID для сохранения порядка
            sorted_items = sorted(vocab_data.items(), key=lambda x: x[1])
            
            # Добавление токенов в наш словарь
            for token, idx in sorted_items:
                if idx < self.vocab_size and token not in self.word_to_idx:
                    self.word_to_idx[token] = idx
                    self.idx_to_word[idx] = token
                    self.vocab[token] = idx
                    
            logger.info(f"Загружено {len(self.word_to_idx)} токенов из DeepSeek словаря")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке DeepSeek словаря: {e}")
            return False
            
    def build_vocab(self, texts: List[str]):
        """Построение словаря из текстов (дополнительно к DeepSeek)"""
        logger.info("Начало построения дополнительного словаря...")
        word_freq = {}
        
        # Сбор частот слов
        for text in texts:
            words = self._preprocess_text(text).split()
            for word in words:
                if word not in self.word_to_idx:  # Только новые слова
                    word_freq[word] = word_freq.get(word, 0) + 1
                    
        # Сортировка по частоте
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Добавление новых слов в словарь
        idx = len(self.word_to_idx)
        for word, freq in sorted_words:
            if idx >= self.vocab_size:
                break
            if word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                self.vocab[word] = freq
                idx += 1
                
        logger.info(f"Дополнительный словарь построен. Общий размер: {len(self.word_to_idx)}")
        
    def _preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        # Приведение к нижнему регистру
        text = text.lower()
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        # Удаление лишних символов (сохраняя пунктуацию)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        # Удаление повторяющихся символов
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
        return text.strip()
        
    def encode(self, text: str) -> List[int]:
        """Кодирование текста в последовательность токенов"""
        # Используем regex паттерн для токенизации
        words = self.compiled_pattern.findall(text)
        tokens = [self.special_tokens['<｜begin▁of▁sentence｜>']]
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                # Попытка разбить на подтокены (простое разбиение)
                subtokens = self._subword_tokenize(word)
                tokens.extend(subtokens)
                
        tokens.append(self.special_tokens['<｜end▁of▁sentence｜>'])
        return tokens
        
    def _subword_tokenize(self, word: str) -> List[int]:
        """Разбиение слова на подтокены"""
        subtokens = []
        
        # Простое разбиение на символы если слово не найдено
        for char in word:
            if char in self.word_to_idx:
                subtokens.append(self.word_to_idx[char])
            else:
                subtokens.append(self.special_tokens['<UNK>'])
                
        return subtokens if subtokens else [self.special_tokens['<UNK>']]
        
    def decode(self, tokens: List[int]) -> str:
        """Декодирование последовательности токенов в текст"""
        words = []
        for token in tokens:
            if token in self.idx_to_word:
                word = self.idx_to_word[token]
                # Пропуск специальных токенов
                if word not in ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜▁pad▁｜>', '<UNK>']:
                    words.append(word)
                    
        return ' '.join(words)

class SinDataset(Dataset):
    """Датасет для обучения чат-бота"""
    
    def __init__(self, texts: List[str], tokenizer: SinTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info("Подготовка датасета...")
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > 1:
                # Создание пар (вход, цель) для обучения
                for i in range(1, len(tokens)):
                    input_seq = tokens[:i]
                    target_seq = tokens[i]
                    
                    # Паддинг
                    if len(input_seq) < max_length:
                        input_seq.extend([tokenizer.special_tokens['<｜▁pad▁｜>']] * 
                                       (max_length - len(input_seq)))
                    else:
                        input_seq = input_seq[:max_length]
                        
                    self.data.append((input_seq, target_seq))
                    
        logger.info(f"Датасет подготовлен. Размер: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class SinModel(nn.Module):
    """Кастомная нейросеть для чат-бота Sin"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512, hidden_dim: int = 1024, 
                 num_layers: int = 4, dropout: float = 0.1):
        super(SinModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Эмбеддинги
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=2)  # PAD token index
        
        # LSTM слои
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Инициализация весов
        self._init_weights()
        
        logger.info(f"Модель создана. Параметры: vocab_size={vocab_size}, "
                   f"embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}")
        
    def _init_weights(self):
        """Инициализация весов"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x, hidden=None):
        """Прямой проход"""
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.output_layer(lstm_out)
        return output, hidden
        
    def get_weights_info(self) -> Dict[str, float]:
        """Получение информации о весах для логгирования"""
        weights_info = {}
        total_params = 0
        total_grad_norm = 0.0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2).item()
                weights_info[f"{name}_norm"] = param_norm
                total_params += param.numel()
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    weights_info[f"{name}_grad_norm"] = grad_norm
                    total_grad_norm += grad_norm
                    
        weights_info["total_parameters"] = total_params
        weights_info["total_gradient_norm"] = total_grad_norm
        
        return weights_info

class TrainingMetrics:
    """Класс для отслеживания метрик обучения"""
    
    def __init__(self):
        self.loss_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        self.perplexity_history = deque(maxlen=1000)
        self.weights_info_history = deque(maxlen=100)
        self.start_time = time.time()
        self.epoch_start_time = None
        self.batch_start_time = None
        self.total_batches = 0
        self.completed_batches = 0
        
    def update_loss(self, loss: float):
        """Обновление метрики потерь"""
        self.loss_history.append(loss)
        
    def update_accuracy(self, accuracy: float):
        """Обновление метрики точности"""
        self.accuracy_history.append(accuracy)
        
    def update_perplexity(self, perplexity: float):
        """Обновление метрики перплексии"""
        self.perplexity_history.append(perplexity)
        
    def update_weights_info(self, weights_info: Dict[str, float]):
        """Обновление информации о весах"""
        self.weights_info_history.append(weights_info)
        
    def start_epoch(self):
        """Начало эпохи"""
        self.epoch_start_time = time.time()
        self.completed_batches = 0
        
    def start_batch(self):
        """Начало батча"""
        self.batch_start_time = time.time()
        
    def end_batch(self):
        """Конец батча"""
        self.completed_batches += 1
        self.total_batches += 1
        
    def get_eta(self, total_batches: int) -> str:
        """Расчет оставшегося времени"""
        if self.completed_batches == 0:
            return "Расчет..."
            
        elapsed_time = time.time() - self.epoch_start_time
        avg_time_per_batch = elapsed_time / self.completed_batches
        remaining_batches = total_batches - self.completed_batches
        eta_seconds = avg_time_per_batch * remaining_batches
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def get_average_loss(self) -> float:
        """Средние потери"""
        return np.mean(self.loss_history) if self.loss_history else 0.0
        
    def get_average_accuracy(self) -> float:
        """Средняя точность"""
        return np.mean(self.accuracy_history) if self.accuracy_history else 0.0
        
    def get_average_perplexity(self) -> float:
        """Средняя перплексия"""
        return np.mean(self.perplexity_history) if self.perplexity_history else 0.0

class SinChatBot:
    """Основной класс чат-бота Sin"""
    
    def __init__(self, model_path: str = "sin_model", vocab_size: int = 128000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.model = None
        self.metrics = TrainingMetrics()
        self.is_training = False
        
        # Создание директории для модели
        os.makedirs(model_path, exist_ok=True)
        
        # Автозагрузка модели и токенизатора
        self.load_model()
        
        # Обработчик сигналов для автосохранения
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info("Получен сигнал завершения. Сохранение модели...")
        self.save_model()
        sys.exit(0)
        
    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            # Загрузка токенизатора
            tokenizer_path = os.path.join(self.model_path, "tokenizer.pkl")
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Токенизатор загружен")
            else:
                self.tokenizer = SinTokenizer(self.vocab_size)
                # Попытка загрузить DeepSeek словарь
                deepseek_vocab_path = os.path.join(self.model_path, "tokenizer.json")
                if os.path.exists(deepseek_vocab_path):
                    try:
                        with open(deepseek_vocab_path, 'r', encoding='utf-8') as f:
                            vocab_data = json.load(f)
                        self.tokenizer.load_deepseek_vocab(vocab_data)
                        logger.info("DeepSeek словарь загружен")
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке DeepSeek словаря: {e}")
                logger.info("Создан новый токенизатор")
                
            # Загрузка модели
            model_path = os.path.join(self.model_path, "model.pth")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.model = SinModel(
                    vocab_size=len(self.tokenizer.word_to_idx),
                    embedding_dim=checkpoint.get('embedding_dim', 512),
                    hidden_dim=checkpoint.get('hidden_dim', 1024),
                    num_layers=checkpoint.get('num_layers', 4)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Модель загружена")
            else:
                self.model = SinModel(vocab_size=len(self.tokenizer.word_to_idx))
                logger.info("Создана новая модель")
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            logger.error(traceback.format_exc())
            # Создание новых объектов в случае ошибки
            self.tokenizer = SinTokenizer(self.vocab_size)
            self.model = SinModel(vocab_size=len(self.tokenizer.word_to_idx))
            
    def save_model(self):
        """Сохранение модели и токенизатора"""
        try:
            # Сохранение токенизатора
            tokenizer_path = os.path.join(self.model_path, "tokenizer.pkl")
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
                
            # Сохранение модели
            model_path = os.path.join(self.model_path, "model.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': len(self.tokenizer.word_to_idx),
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            }, model_path)
            
            logger.info("Модель и токенизатор сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            logger.error(traceback.format_exc())
            
    def parse_url(self, url: str) -> str:
        """Парсинг URL и извлечение текста"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Удаление скриптов и стилей
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Извлечение текста
            text = soup.get_text()
            
            # Очистка текста
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Текст извлечен из URL: {url}")
            return text
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге URL {url}: {e}")
            logger.error(traceback.format_exc())
            return ""
            
    def read_txt_file(self, file_path: str) -> str:
        """Чтение TXT файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"TXT файл прочитан: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Ошибка при чтении TXT файла {file_path}: {e}")
            logger.error(traceback.format_exc())
            return ""
            
    def read_docx_file(self, file_path: str) -> str:
        """Чтение DOCX файла"""
        try:
            doc = Document(file_path)
            content = []
            for paragraph in doc.paragraphs:
                content.append(paragraph.text)
            text = '\n'.join(content)
            logger.info(f"DOCX файл прочитан: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении DOCX файла {file_path}: {e}")
            logger.error(traceback.format_exc())
            return ""
            
    def read_json_file(self, file_path: str) -> str:
        """Чтение JSON файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Преобразование JSON в текст
            def json_to_text(obj, level=0):
                text = ""
                indent = "  " * level
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        text += f"{indent}{key}: {json_to_text(value, level + 1)}\n"
                elif isinstance(obj, list):
                    for item in obj:
                        text += f"{indent}- {json_to_text(item, level + 1)}\n"
                else:
                    text += str(obj)
                return text
                
            text = json_to_text(data)
            logger.info(f"JSON файл прочитан: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении JSON файла {file_path}: {e}")
            logger.error(traceback.format_exc())
            return ""
            
    def read_pdf_file(self, file_path: str) -> str:
        """Чтение PDF файла"""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            logger.info(f"PDF файл прочитан: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении PDF файла {file_path}: {e}")
            logger.error(traceback.format_exc())
            return ""
            
    def process_file(self, file_path: str) -> str:
        """Обработка файла в зависимости от расширения"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.txt':
            return self.read_txt_file(file_path)
        elif ext == '.docx':
            return self.read_docx_file(file_path)
        elif ext == '.json':
            return self.read_json_file(file_path)
        elif ext == '.pdf':
            return self.read_pdf_file(file_path)
        else:
            logger.warning(f"Неподдерживаемый формат файла: {file_path}")
            return ""
            
    def prepare_training_data(self, sources: List[Dict[str, str]]) -> List[str]:
        """Подготовка данных для обучения"""
        texts = []
        
        for source in sources:
            try:
                if 'url' in source:
                    text = self.parse_url(source['url'])
                    if text:
                        texts.append(text)
                        
                elif 'file' in source:
                    text = self.process_file(source['file'])
                    if text:
                        texts.append(text)
                        
                elif 'text' in source:
                    texts.append(source['text'])
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке источника {source}: {e}")
                logger.error(traceback.format_exc())
                
        return texts
        
    def calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Расчет точности"""
        predicted_tokens = torch.argmax(predictions, dim=-1)
        correct = (predicted_tokens == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0
        
    def calculate_perplexity(self, loss: float) -> float:
        """Расчет перплексии"""
        return math.exp(loss) if loss < 100 else float('inf')
        
    def train_on_data(self, sources: List[Dict[str, str]], epochs: int = 5, 
                     batch_size: int = 32, learning_rate: float = 0.001):
        """Обучение на данных"""
        try:
            self.is_training = True
            logger.info("Начало подготовки данных для обучения...")
            
            # Подготовка данных
            texts = self.prepare_training_data(sources)
            if not texts:
                logger.warning("Нет данных для обучения")
                return
                
            # Обновление словаря токенизатора
            logger.info("Обновление словаря токенизатора...")
            self.tokenizer.build_vocab(texts)
            
            # Создание или обновление модели
            vocab_size = len(self.tokenizer.word_to_idx)
            if self.model is None or self.model.vocab_size != vocab_size:
                self.model = SinModel(vocab_size=vocab_size)
                logger.info("Создана новая модель с обновленным словарем")
                
            # Создание датасета
            dataset = SinDataset(texts, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Настройка оптимизатора и функции потерь
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<｜▁pad▁｜>'])
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Обучение
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.train()
            
            logger.info(f"Начало обучения. Эпохи: {epochs}, Батчей: {len(dataloader)}")
            
            for epoch in range(epochs):
                self.metrics.start_epoch()
                total_loss = 0.0
                total_accuracy = 0.0
                total_perplexity = 0.0
                
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    self.metrics.start_batch()
                    
                    try:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # Прямой проход
                        optimizer.zero_grad()
                        outputs, _ = self.model(inputs)
                        
                        # Расчет потерь
                        loss = criterion(outputs[:, -1, :], targets)
                        
                        # Обратный проход
                        loss.backward()
                        optimizer.step()
                        
                        # Расчет метрик
                        accuracy = self.calculate_accuracy(outputs[:, -1, :], targets)
                        perplexity = self.calculate_perplexity(loss.item())
                        
                        # Обновление метрик
                        self.metrics.update_loss(loss.item())
                        self.metrics.update_accuracy(accuracy)
                        self.metrics.update_perplexity(perplexity)
                        
                        total_loss += loss.item()
                        total_accuracy += accuracy
                        total_perplexity += perplexity
                        
                        # Получение информации о весах
                        weights_info = self.model.get_weights_info()
                        self.metrics.update_weights_info(weights_info)
                        
                        # Логгирование
                        if batch_idx % 10 == 0:
                            avg_loss = total_loss / (batch_idx + 1)
                            avg_accuracy = total_accuracy / (batch_idx + 1)
                            avg_perplexity = total_perplexity / (batch_idx + 1)
                            eta = self.metrics.get_eta(len(dataloader))
                            
                            logger.info(f"Эпоха {epoch+1}/{epochs}, "
                                      f"Батч {batch_idx+1}/{len(dataloader)}, "
                                      f"Потери: {avg_loss:.4f}, "
                                      f"Точность: {avg_accuracy:.4f}, "
                                      f"Перплексия: {avg_perplexity:.4f}, "
                                      f"ETA: {eta}")
                            
                            # Логгирование информации о весах
                            if weights_info:
                                logger.info(f"Норма градиентов: {weights_info.get('total_gradient_norm', 0):.6f}")
                                
                    except Exception as e:
                        logger.error(f"Ошибка в батче {batch_idx}: {e}")
                        logger.error(traceback.format_exc())
                        continue
                        
                    finally:
                        self.metrics.end_batch()
                        
                # Конец эпохи
                avg_epoch_loss = total_loss / len(dataloader)
                avg_epoch_accuracy = total_accuracy / len(dataloader)
                avg_epoch_perplexity = total_perplexity / len(dataloader)
                
                logger.info(f"Эпоха {epoch+1} завершена. "
                          f"Средние потери: {avg_epoch_loss:.4f}, "
                          f"Точность: {avg_epoch_accuracy:.4f}, "
                          f"Перплексия: {avg_epoch_perplexity:.4f}")
                          
                # Сохранение промежуточной модели
                self.save_model()
                
            logger.info("Обучение завершено")
            
        except Exception as e:
            logger.error(f"Ошибка во время обучения: {e}")
            logger.error(traceback.format_exc())
            
        finally:
            self.is_training = False
            self.save_model()
            
    def train_on_dialogue(self, user_input: str, bot_response: str):
        """Обучение на диалоге"""
        try:
            # Создание текста диалога
            dialogue_text = f"Пользователь: {user_input}\nБот: {bot_response}"
            
            # Обновление токенизатора
            self.tokenizer.build_vocab([dialogue_text])
            
            # Обновление модели
            vocab_size = len(self.tokenizer.word_to_idx)
            if self.model is None or self.model.vocab_size != vocab_size:
                self.model = SinModel(vocab_size=vocab_size)
                
            # Подготовка данных для обучения
            dataset = SinDataset([dialogue_text], self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Настройка обучения
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<｜▁pad▁｜>'])
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.train()
            
            # Обучение на одном примере
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = criterion(outputs[:, -1, :], targets)
                loss.backward()
                optimizer.step()
                
                logger.info(f"Модель обучена на диалоге. Потери: {loss.item():.4f}")
                break
                
        except Exception as e:
            logger.error(f"Ошибка при обучении на диалоге: {e}")
            logger.error(traceback.format_exc())
            
    def generate_response(self, prompt: str, max_length: int = 100, 
                         temperature: float = 0.8) -> str:
        """Генерация ответа"""
        try:
            if self.model is None or self.tokenizer is None:
                return "Извините, модель еще не обучена."
                
            self.model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            
            # Кодирование входного текста
            input_tokens = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            
            # Генерация
            generated_tokens = input_tokens.copy()
            
            with torch.no_grad():
                hidden = None
                for _ in range(max_length):
                    # Получение последнего токена
                    current_input = input_tensor[:, -1].unsqueeze(0)
                    
                    # Прогноз
                    output, hidden = self.model(current_input, hidden)
                    predictions = output[0, -1, :]
                    
                    # Применение температуры
                    predictions = predictions / temperature
                    probabilities = torch.softmax(predictions, dim=-1)
                    
                    # Сэмплинг
                    next_token = torch.multinomial(probabilities, 1).item()
                    
                    # Проверка на специальные токены
                    if next_token in [self.tokenizer.special_tokens['<｜end▁of▁sentence｜>'], 
                                    self.tokenizer.special_tokens['<｜▁pad▁｜>']]:
                        break
                        
                    generated_tokens.append(next_token)
                    input_tensor = torch.cat([input_tensor, 
                                            torch.tensor([[next_token]], 
                                                       dtype=torch.long).to(device)], dim=1)
                    
            # Декодирование
            response = self.tokenizer.decode(generated_tokens[len(input_tokens):])
            return response.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            logger.error(traceback.format_exc())
            return "Извините, произошла ошибка при генерации ответа."

def main():
    """Основная функция для демонстрации работы"""
    # Создание чат-бота
    bot = SinChatBot(model_path="sin_model")
    
    print("Добро пожаловать в чат-бот Sin!")
    print("Доступные команды:")
    print("1. /train - обучение на данных")
    print("2. /dialogue - обучение на диалоге")
    print("3. /chat - режим чата")
    print("4. /exit - выход")
    
    while True:
        try:
            command = input("\nВведите команду: ").strip().lower()
            
            if command == '/exit':
                print("Сохранение модели и выход...")
                bot.save_model()
                break
                
            elif command == '/train':
                print("Выберите источник данных:")
                print("1. URL")
                print("2. Файл")
                print("3. Текст")
                
                choice = input("Выбор (1-3): ").strip()
                sources = []
                
                if choice == '1':
                    url = input("Введите URL: ").strip()
                    sources.append({'url': url})
                elif choice == '2':
                    file_path = input("Введите путь к файлу: ").strip()
                    sources.append({'file': file_path})
                elif choice == '3':
                    text = input("Введите текст: ").strip()
                    sources.append({'text': text})
                else:
                    print("Неверный выбор")
                    continue
                    
                epochs = int(input("Количество эпох (по умолчанию 5): ") or "5")
                bot.train_on_data(sources, epochs=epochs)
                
            elif command == '/dialogue':
                user_input = input("Ваше сообщение: ").strip()
                bot_response = input("Ответ бота: ").strip()
                bot.train_on_dialogue(user_input, bot_response)
                print("Модель обучена на диалоге")
                
            elif command == '/chat':
                print("Режим чата. Введите /back для возврата в главное меню")
                while True:
                    user_input = input("Вы: ").strip()
                    if user_input == '/back':
                        break
                    response = bot.generate_response(user_input)
                    print(f"Бот: {response}")
                    
            else:
                print("Неизвестная команда")
                
        except KeyboardInterrupt:
            print("\nСохранение модели и выход...")
            bot.save_model()
            break
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            logger.error(traceback.format_exc())
            print("Произошла ошибка. Проверьте логи.")

if __name__ == "__main__":
    main()
