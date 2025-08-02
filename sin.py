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
from collections import deque, OrderedDict
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
import ast

# Попытка импортировать transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Заглушка для AutoTokenizer, если transformers не установлены
    class AutoTokenizer:
        pass

# Настройка логгирования
# Установка рабочей директории
PROJECT_DIR = r"C:\Users\User\Downloads\SinChatBot"
MODEL_DIR = os.path.join(PROJECT_DIR, "sin_model")
LOG_FILE = os.path.join(PROJECT_DIR, "sin_chatbot.log")
DEEPSEEK_VOCAB_FILE = os.path.join(MODEL_DIR, "tokenizer.json")

# Создание необходимых директорий
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if TRANSFORMERS_AVAILABLE:
    logger.info("Библиотека transformers доступна.")
else:
    logger.warning("Библиотека transformers не найдена. Будет использован кастомный токенизатор.")

# --- Кастомный токенизатор (резервный вариант) ---
class SinTokenizer:
    """Кастомный токенизатор для чат-бота Sin с поддержкой DeepSeek словаря"""
    
    def __init__(self, vocab_size: int = 128000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = {}
        
        # Инициализируем базовые специальные токены. Они могут быть перезаписаны при загрузке словаря.
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
            # Добавляем только если токен еще не в словаре или имеет другой индекс
            if token not in self.word_to_idx or self.word_to_idx[token] != idx:
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
            
    def load_deepseek_vocab(self, vocab_data_or_path) -> bool:
        """Загрузка словаря из данных DeepSeek или пути к файлу"""
        try:
            logger.info("Начало загрузки словаря DeepSeek...")
            
            # Если передан путь к файлу, попробуем прочитать его
            if isinstance(vocab_data_or_path, str):
                vocab_file_path = vocab_data_or_path
                vocab_data = None
                try:
                    with open(vocab_file_path, 'r', encoding='utf-8') as f:
                        vocab_data = json.load(f)
                    logger.info("Файл успешно загружен как стандартный JSON")
                except json.JSONDecodeError as je:
                    logger.error(f"Не удалось загрузить как стандартный JSON: {je}")
                    return False
            else:
                # Предполагаем, что это уже словарь
                vocab_data = vocab_data_or_path

            if not isinstance(vocab_data, dict):
                logger.error("Загруженные данные не являются словарем")
                return False

            # Проверяем структуру файла. 
            # Если есть ключ 'added_tokens', это формат токенизатора Hugging Face
            if 'added_tokens' in vocab_data and isinstance(vocab_data['added_tokens'], list):
                logger.info("Обнаружен формат токенизатора Hugging Face (added_tokens)")
                # Извлекаем токены из массива added_tokens
                added_tokens_list = vocab_data['added_tokens']
                token_dict = {}
                loaded_special_tokens = {}
                
                for token_info in added_tokens_list:
                    if isinstance(token_info, dict):
                        token_id = token_info.get('id')
                        token_content = token_info.get('content')
                        is_special = token_info.get('special', False)
                        
                        if isinstance(token_id, int) and isinstance(token_content, str):
                            token_dict[token_content] = token_id
                            if is_special or token_content in ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜▁pad▁｜>', '<UNK>'] or token_content.startswith('<｜place▁holder'):
                                loaded_special_tokens[token_content] = token_id
                                
                # Также проверяем основной словарь vocab, если он есть
                if 'model' in vocab_data and 'vocab' in vocab_data['model'] and isinstance(vocab_data['model']['vocab'], dict):
                    logger.info("Найден основной словарь vocab в model")
                    # Добавляем обычные токены из vocab
                    for token, idx in vocab_data['model']['vocab'].items():
                        if isinstance(idx, int) and isinstance(token, str) and token not in token_dict:
                            token_dict[token] = idx
                            
            # Если есть ключ 'model' с подключом 'vocab', это тоже формат токенизатора
            elif 'model' in vocab_data and 'vocab' in vocab_data['model'] and isinstance(vocab_data['model']['vocab'], dict):
                logger.info("Обнаружен формат токенизатора (model.vocab)")
                token_dict = vocab_data['model']['vocab']
                loaded_special_tokens = {}
                # Определяем специальные токены по имени
                for token, idx in token_dict.items():
                    if isinstance(idx, int) and isinstance(token, str):
                        if token in ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜▁pad▁｜>', '<UNK>'] or token.startswith('<｜place▁holder'):
                            loaded_special_tokens[token] = idx
            # Иначе предполагаем, что это простой словарь
            else:
                logger.info("Обнаружен формат простого словаря")
                token_dict = vocab_data
                loaded_special_tokens = {}
                # Определяем специальные токены по имени
                for token, idx in token_dict.items():
                    if isinstance(idx, int) and isinstance(token, str):
                        if token in ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜▁pad▁｜>', '<UNK>'] or token.startswith('<｜place▁holder'):
                            loaded_special_tokens[token] = idx

            # Фильтрация и проверка типов данных
            filtered_items = []
            for token, idx in token_dict.items():
                if isinstance(idx, int) and isinstance(token, str):
                    filtered_items.append((token, idx))
                else:
                    logger.debug(f"Пропущен некорректный элемент словаря: {token}: {idx}")
            
            # Если в загруженном словаре есть специальные токены, обновляем наш список
            if loaded_special_tokens:
                logger.info(f"Найдены специальные токены в загруженном словаре: {list(loaded_special_tokens.keys())}")
                self.special_tokens.update(loaded_special_tokens)
            
            # Сортировка по ID для сохранения порядка
            sorted_items = sorted(filtered_items, key=lambda x: x[1])
            
            # Очищаем существующие словари перед загрузкой
            self.word_to_idx.clear()
            self.idx_to_word.clear()
            self.vocab.clear()
            
            # Первоначально добавляем специальные токены
            self._initialize_special_tokens()
            
            # Добавление токенов в наши словари
            added_count = 0
            for token, idx in sorted_items:
                # Проверяем, не выходит ли индекс за пределы vocab_size
                if idx < self.vocab_size:
                    self.word_to_idx[token] = idx
                    self.idx_to_word[idx] = token
                    self.vocab[token] = idx
                    added_count += 1
                # else:
                #     logger.debug(f"Токен {token} (ID: {idx}) превышает лимит vocab_size ({self.vocab_size}) и пропущен.")
                    
            logger.info(f"Загружено {added_count} токенов из DeepSeek словаря (всего в файле: {len(sorted_items)})")
            logger.info(f"Финальные специальные токены: {self.special_tokens}")
            logger.debug(f"BOS ID: {self.special_tokens.get('<｜begin▁of▁sentence｜>', 'Not found')}")
            logger.debug(f"EOS ID: {self.special_tokens.get('<｜end▁of▁sentence｜>', 'Not found')}")
            logger.debug(f"PAD ID: {self.special_tokens.get('<｜▁pad▁｜>', 'Not found')}")
            logger.debug(f"UNK ID: {self.special_tokens.get('<UNK>', 'Not found')}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке DeepSeek словаря: {e}")
            logger.error(traceback.format_exc())
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
        # Начинаем с индекса после максимального существующего
        start_idx = max(self.word_to_idx.values()) + 1 if self.word_to_idx else len(self.special_tokens)
        idx = start_idx
        
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
        words = self._preprocess_text(text).split()
        # Используем ID из словаря special_tokens для BOS
        bos_id = self.special_tokens.get('<｜begin▁of▁sentence｜>', 0)
        tokens = [bos_id]
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                # Попытка разбить на подтокены (простое разбиение)
                subtokens = self._subword_tokenize(word)
                tokens.extend(subtokens)
                
        # Добавляем EOS токен
        eos_id = self.special_tokens.get('<｜end▁of▁sentence｜>', 1)
        tokens.append(eos_id)
        return tokens
        
    def _subword_tokenize(self, word: str) -> List[int]:
        """Разбиение слова на подтокены"""
        subtokens = []
        
        # Простое разбиение на символы если слово не найдено
        for char in word:
            if char in self.word_to_idx:
                subtokens.append(self.word_to_idx[char])
            else:
                # Используем ID из словаря special_tokens для UNK
                unk_id = self.special_tokens.get('<UNK>', 3)
                subtokens.append(unk_id)
                
        return subtokens if subtokens else [self.special_tokens.get('<UNK>', 3)]
        
    def decode(self, tokens: List[int]) -> str:
        """Декодирование последовательности токенов в текст"""
        words = []
        special_token_ids = set(self.special_tokens.values())
        
        for token in tokens:
            if token in self.idx_to_word and token not in special_token_ids:
                word = self.idx_to_word[token]
                # Пропуск специальных токенов
                if word not in ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜▁pad▁｜>', '<UNK>']:
                    # Убираем специальные символы из вывода, если они есть
                    if not word.startswith('<｜place▁holder'):
                        words.append(word)
                    
        return ' '.join(words)

# --- Классы модели и датасета ---
class SinDataset(Dataset):
    """Датасет для обучения чат-бота"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info("Подготовка датасета...")
        for text in texts:
            # Проверяем, является ли токенизатор токенизатором из transformers
            if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, AutoTokenizer):
                # Используем универсальный метод encode для transformers
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0) # [seq_len]
            else:
                # Для кастомного токенизатора
                tokens = self.tokenizer.encode(text)
                # Паддинг/обрезка вручную
                if len(tokens) < self.max_length:
                    # Определяем pad_token_id
                    pad_token_id = self.tokenizer.special_tokens.get('<｜▁pad▁｜>', 2) if hasattr(self.tokenizer, 'special_tokens') else 0
                    tokens.extend([pad_token_id] * (self.max_length - len(tokens)))
                else:
                    tokens = tokens[:self.max_length]
                input_ids = torch.tensor(tokens, dtype=torch.long)
            
            # Создание пар (вход, цель) для обучения
            # Для каждого токена i, вход - это токены от 0 до i-1, цель - токен i
            for i in range(1, len(input_ids)):
                input_seq = input_ids[:i]
                target_seq = input_ids[i]
                
                # Паддинг входной последовательности до max_length
                if len(input_seq) < max_length:
                    # Определяем pad_token_id
                    if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, AutoTokenizer):
                        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else 0
                    else:
                        # Для кастомного токенизатора
                        pad_token_id = getattr(self.tokenizer, 'special_tokens', {}).get('<｜▁pad▁｜>', 2)
                    input_seq = torch.cat([input_seq, torch.full((max_length - len(input_seq),), pad_token_id, dtype=torch.long)])
                else:
                    input_seq = input_seq[:max_length]
                    
                self.data.append((input_seq, target_seq))
                    
        logger.info(f"Датасет подготовлен. Размер: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return input_seq, target_seq

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
        # pad_token_id будет определен позже, при создании модели
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
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
    
    def __init__(self, model_path: str = MODEL_DIR, vocab_size: int = 128000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.model = None
        self.metrics = TrainingMetrics()
        self.is_training = False
        
        # Проверка наличия необходимых файлов и папок
        self._check_files_and_dirs()
        
        # Автозагрузка модели и токенизатора
        self.load_model()
        
        # Обработчик сигналов для автосохранения
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _check_files_and_dirs(self):
        """Проверка наличия необходимых файлов и папок"""
        logger.info("Проверка файлов и папок...")
        
        # Проверка основной директории
        if not os.path.exists(PROJECT_DIR):
            os.makedirs(PROJECT_DIR)
            logger.info(f"Создана директория проекта: {PROJECT_DIR}")
            
        # Проверка директории модели
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            logger.info(f"Создана директория модели: {MODEL_DIR}")
            
        # Проверка лог-файла
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w') as f:
                f.write("")
            logger.info(f"Создан лог-файл: {LOG_FILE}")
            
        logger.info("Проверка файлов и папок завершена")
        
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
            
            # Попробуем загрузить сохраненный токенизатор
            if os.path.exists(tokenizer_path) and not TRANSFORMERS_AVAILABLE:
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Кастомный токенизатор загружен")
            elif TRANSFORMERS_AVAILABLE and os.path.exists(DEEPSEEK_VOCAB_FILE):
                # Используем AutoTokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                    logger.info("Оригинальный токенизатор DeepSeek загружен через transformers")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить токенизатор через transformers: {e}")
                    logger.warning("Используется кастомный токенизатор как резервный вариант")
                    self.tokenizer = SinTokenizer(self.vocab_size)
                    if self.tokenizer.load_deepseek_vocab(DEEPSEEK_VOCAB_FILE):
                        logger.info("DeepSeek словарь успешно загружен в кастомный токенизатор")
                    else:
                        logger.warning("Не удалось загрузить DeepSeek словарь")
            else:
                # Создаем кастомный токенизатор
                self.tokenizer = SinTokenizer(self.vocab_size)
                if os.path.exists(DEEPSEEK_VOCAB_FILE):
                    if self.tokenizer.load_deepseek_vocab(DEEPSEEK_VOCAB_FILE):
                        logger.info("DeepSeek словарь успешно загружен")
                    else:
                        logger.warning("Не удалось загрузить DeepSeek словарь")
                logger.info("Создан новый токенизатор")
                
            # Определяем размер словаря
            if hasattr(self.tokenizer, 'vocab_size'):
                vocab_size = self.tokenizer.vocab_size
            elif hasattr(self.tokenizer, 'get_vocab'):
                vocab_size = len(self.tokenizer.get_vocab())
            else:
                vocab_size = len(getattr(self.tokenizer, 'word_to_idx', {}))
                
            logger.info(f"Размер словаря токенизатора: {vocab_size}")

            # Загрузка модели
            model_path = os.path.join(self.model_path, "model.pth")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.model = SinModel(
                    vocab_size=vocab_size,
                    embedding_dim=checkpoint.get('embedding_dim', 512),
                    hidden_dim=checkpoint.get('hidden_dim', 1024),
                    num_layers=checkpoint.get('num_layers', 4)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Модель загружена")
            else:
                self.model = SinModel(vocab_size=vocab_size)
                logger.info("Создана новая модель")
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            logger.error(traceback.format_exc())
            # Создание новых объектов в случае ошибки
            if TRANSFORMERS_AVAILABLE and os.path.exists(DEEPSEEK_VOCAB_FILE):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                except:
                    self.tokenizer = SinTokenizer(self.vocab_size)
            else:
                self.tokenizer = SinTokenizer(self.vocab_size)
            
            vocab_size = len(getattr(self.tokenizer, 'word_to_idx', {})) if not TRANSFORMERS_AVAILABLE else 128000
            self.model = SinModel(vocab_size=vocab_size)
            
    def save_model(self):
        """Сохранение модели и токенизатора"""
        try:
            # Сохранение токенизатора (только если это кастомный токенизатор)
            if not TRANSFORMERS_AVAILABLE or not isinstance(self.tokenizer, AutoTokenizer):
                tokenizer_path = os.path.join(self.model_path, "tokenizer.pkl")
                with open(tokenizer_path, 'wb') as f:
                    pickle.dump(self.tokenizer, f)
                
            # Сохранение модели
            model_path = os.path.join(self.model_path, "model.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': len(getattr(self.tokenizer, 'word_to_idx', {})) if not TRANSFORMERS_AVAILABLE else (
                    self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else len(self.tokenizer.get_vocab())
                ),
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
            logger.info("=" * 60)
            logger.info("Начало подготовки данных для обучения...")
            logger.info("=" * 60)
            
            # Подготовка данных
            texts = self.prepare_training_data(sources)
            if not texts:
                logger.warning("Нет данных для обучения")
                return
                
            # Для кастомного токенизатора обновляем словарь
            if not TRANSFORMERS_AVAILABLE or not isinstance(self.tokenizer, AutoTokenizer):
                logger.info("Обновление словаря кастомного токенизатора...")
                self.tokenizer.build_vocab(texts)
            
            # Определяем размер словаря
            if hasattr(self.tokenizer, 'vocab_size'):
                vocab_size = self.tokenizer.vocab_size
            elif hasattr(self.tokenizer, 'get_vocab'):
                vocab_size = len(self.tokenizer.get_vocab())
            else:
                vocab_size = len(getattr(self.tokenizer, 'word_to_idx', {}))
                
            # Создание или обновление модели
            if self.model is None or self.model.vocab_size != vocab_size:
                logger.info(f"Создание новой модели с vocab_size={vocab_size}")
                self.model = SinModel(vocab_size=vocab_size)
                logger.info("Создана новая модель с обновленным словарем")
            else:
                logger.info("Используется существующая модель")
                
            # Создание датасета
            dataset = SinDataset(texts, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Определяем pad_token_id для функции потерь
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                # Для кастомного токенизатора
                pad_token_id = getattr(self.tokenizer, 'special_tokens', {}).get('<｜▁pad▁｜>', 2)
                
            # Настройка оптимизатора и функции потерь
            criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Обучение
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.train()
            
            logger.info(f"Начало обучения.")
            logger.info(f"  Эпохи: {epochs}")
            logger.info(f"  Батчей: {len(dataloader)}")
            logger.info(f"  Размер батча: {batch_size}")
            logger.info(f"  Learning rate: {learning_rate}")
            logger.info(f"  Устройство: {device}")
            logger.info("=" * 60)
            
            for epoch in range(epochs):
                self.metrics.start_epoch()
                total_loss = 0.0
                total_accuracy = 0.0
                total_perplexity = 0.0
                
                logger.info(f"ЭПОХА {epoch+1}/{epochs} НАЧАЛАСЬ")
                logger.info("-" * 40)
                
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
                        
                        # Логгирование каждые 10 батчей
                        if batch_idx % 10 == 0:
                            avg_loss = total_loss / (batch_idx + 1)
                            avg_accuracy = total_accuracy / (batch_idx + 1)
                            avg_perplexity = total_perplexity / (batch_idx + 1)
                            eta = self.metrics.get_eta(len(dataloader))
                            
                            logger.info(f"  Батч {batch_idx+1}/{len(dataloader)} | "
                                      f"Потери: {loss.item():.4f} (средние: {avg_loss:.4f}) | "
                                      f"Точность: {accuracy:.4f} (средние: {avg_accuracy:.4f}) | "
                                      f"Перплексия: {perplexity:.4f} (средние: {avg_perplexity:.4f}) | "
                                      f"ETA: {eta}")
                            
                            # Логгирование информации о весах (только для первых батчей, чтобы не засорять лог)
                            if weights_info and batch_idx < 50:
                                grad_norm = weights_info.get('total_gradient_norm', 0)
                                logger.debug(f"    Норма градиентов: {grad_norm:.6f}")
                                
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
                
                logger.info("-" * 40)
                logger.info(f"ЭПОХА {epoch+1}/{epochs} ЗАВЕРШЕНА")
                logger.info(f"  Средние потери: {avg_epoch_loss:.4f}")
                logger.info(f"  Средняя точность: {avg_epoch_accuracy:.4f}")
                logger.info(f"  Средняя перплексия: {avg_epoch_perplexity:.4f}")
                logger.info("=" * 60)
                          
                # Сохранение промежуточной модели
                self.save_model()
                
            logger.info("Обучение завершено")
            logger.info("=" * 60)
            
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
            
            # Для кастомного токенизатора обновляем словарь
            if not TRANSFORMERS_AVAILABLE or not isinstance(self.tokenizer, AutoTokenizer):
                self.tokenizer.build_vocab([dialogue_text])
            
            # Определяем размер словаря
            if hasattr(self.tokenizer, 'vocab_size'):
                vocab_size = self.tokenizer.vocab_size
            elif hasattr(self.tokenizer, 'get_vocab'):
                vocab_size = len(self.tokenizer.get_vocab())
            else:
                vocab_size = len(getattr(self.tokenizer, 'word_to_idx', {}))
                
            # Обновление модели
            if self.model is None or self.model.vocab_size != vocab_size:
                self.model = SinModel(vocab_size=vocab_size)
                
            # Подготовка данных для обучения
            dataset = SinDataset([dialogue_text], self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Определяем pad_token_id для функции потерь
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                # Для кастомного токенизатора
                pad_token_id = getattr(self.tokenizer, 'special_tokens', {}).get('<｜▁pad▁｜>', 2)
                
            # Настройка обучения
            criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.train()
            
            # Обучение на одном примере
            logger.info("Начало обучения на диалоге...")
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = criterion(outputs[:, -1, :], targets)
                loss.backward()
                optimizer.step()
                
                accuracy = self.calculate_accuracy(outputs[:, -1, :], targets)
                perplexity = self.calculate_perplexity(loss.item())
                
                logger.info(f"Модель обучена на диалоге. "
                          f"Потери: {loss.item():.4f}, "
                          f"Точность: {accuracy:.4f}, "
                          f"Перплексия: {perplexity:.4f}")
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
            # Проверяем, является ли токенизатор токенизатором из transformers (который callable)
            if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, AutoTokenizer):
                # Для transformers токенизатора
                encoding = self.tokenizer(
                    prompt,
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
            else:
                # Для кастомного токенизатора или других
                input_tokens = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)
            
            # Генерация
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                hidden = None
                for _ in range(max_length):
                    # Получение последнего токена
                    current_input = generated_ids[:, -1].unsqueeze(0)
                    
                    # Прогноз
                    output, hidden = self.model(current_input, hidden)
                    predictions = output[0, -1, :]
                    
                    # Применение температуры
                    predictions = predictions / temperature
                    probabilities = torch.softmax(predictions, dim=-1)
                    
                    # Сэмплинг
                    next_token = torch.multinomial(probabilities, 1).item()
                    
                    # Проверка на специальные токены
                    # Определяем eos_token_id
                    if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                        eos_token_id = self.tokenizer.eos_token_id
                    else:
                        # Для кастомного токенизатора
                        eos_token_id = getattr(self.tokenizer, 'special_tokens', {}).get('<｜end▁of▁sentence｜>', 1)
                        
                    if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                        pad_token_id = self.tokenizer.pad_token_id
                    else:
                        # Для кастомного токенизатора
                        pad_token_id = getattr(self.tokenizer, 'special_tokens', {}).get('<｜▁pad▁｜>', 2)
                    
                    if next_token in [eos_token_id, pad_token_id]:
                        break
                        
                    # Добавляем новый токен к сгенерированной последовательности
                    generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]], device=device)], dim=1)
                    
            # Декодирование
            if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, AutoTokenizer):
                # Для transformers токенизатора
                response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                # Для кастомного токенизатора
                response = self.tokenizer.decode(generated_ids[0].tolist())
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            logger.error(traceback.format_exc())
            return "Извините, произошла ошибка при генерации ответа."
            
    def get_model_report(self) -> str:
        """Генерация подробного отчета о модели"""
        try:
            if self.model is None:
                return "Модель не загружена"
                
            report = []
            report.append("=" * 50)
            report.append("ОТЧЕТ О СОСТОЯНИИ МОДЕЛИ")
            report.append("=" * 50)
            report.append(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Общая информация о модели
            report.append("1. ОБЩАЯ ИНФОРМАЦИЯ О МОДЕЛИ")
            report.append("-" * 30)
            report.append(f"Размер словаря: {self.model.vocab_size}")
            report.append(f"Размерность эмбеддингов: {self.model.embedding_dim}")
            report.append(f"Размерность скрытого слоя: {self.model.hidden_dim}")
            report.append(f"Количество LSTM слоев: {self.model.num_layers}")
            
            # Подсчет параметров
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            report.append(f"Общее количество параметров: {total_params:,}")
            report.append(f"Обучаемые параметры: {trainable_params:,}")
            report.append("")
            
            # Информация о токенизаторе
            report.append("2. ИНФОРМАЦИЯ О ТОКЕНИЗАТОРЕ")
            report.append("-" * 30)
            if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, AutoTokenizer):
                report.append("Тип токенизатора: Оригинальный DeepSeek (AutoTokenizer)")
                try:
                    vocab_size = len(self.tokenizer.get_vocab())
                    report.append(f"Размер словаря токенов: {vocab_size}")
                except:
                    report.append("Размер словаря токенов: Не удалось определить")
            else:
                report.append("Тип токенизатора: Кастомный (SinTokenizer)")
                report.append(f"Размер словаря токенов: {len(getattr(self.tokenizer, 'word_to_idx', {}))}")
                report.append(f"Специальные токены: {getattr(self.tokenizer, 'special_tokens', {})}")
            report.append("")
            
            # Метрики обучения
            report.append("3. МЕТРИКИ ОБУЧЕНИЯ")
            report.append("-" * 30)
            if self.metrics.loss_history:
                report.append(f"Средние потери: {self.metrics.get_average_loss():.4f}")
                report.append(f"Средняя точность: {self.metrics.get_average_accuracy():.4f}")
                report.append(f"Средняя перплексия: {self.metrics.get_average_perplexity():.4f}")
            else:
                report.append("Нет данных для отображения метрик")
            report.append("")
            
            # Информация о весах
            report.append("4. ИНФОРМАЦИЯ О ВЕСАХ")
            report.append("-" * 30)
            weights_info = self.model.get_weights_info()
            if weights_info:
                report.append(f"Общая норма градиентов: {weights_info.get('total_gradient_norm', 0):.6f}")
                report.append(f"Общее количество параметров: {weights_info.get('total_parameters', 0):,}")
                
                # Подробная информация о слоях
                report.append("\nПодробная информация о слоях:")
                for key, value in weights_info.items():
                    if key not in ['total_parameters', 'total_gradient_norm']:
                        if 'grad_norm' in key:
                            report.append(f"  {key}: {value:.6f}")
                        elif 'norm' in key:
                            report.append(f"  {key}: {value:.6f}")
            else:
                report.append("Нет данных о весах")
            report.append("")
            
            # История обучения
            report.append("5. ИСТОРИЯ ОБУЧЕНИЯ")
            report.append("-" * 30)
            if self.metrics.loss_history:
                # Экранируем фигурные скобки внутри f-строки для списков
                last_5_losses = [f'{x:.4f}' for x in list(self.metrics.loss_history)[-5:]]
                last_5_accuracies = [f'{x:.4f}' for x in list(self.metrics.accuracy_history)[-5:]]
                last_5_perplexities = [f'{x:.4f}' for x in list(self.metrics.perplexity_history)[-5:]]
                
                report.append(f"Последние 5 значений потерь: {last_5_losses}")
                report.append(f"Последние 5 значений точности: {last_5_accuracies}")
                report.append(f"Последние 5 значений перплексии: {last_5_perplexities}")
            else:
                report.append("Нет истории обучения")
            report.append("")
            
            # Рекомендации
            report.append("6. РЕКОМЕНДАЦИИ")
            report.append("-" * 30)
            if self.metrics.loss_history:
                avg_loss = self.metrics.get_average_loss()
                if avg_loss > 2.0:
                    report.append("• Рекомендуется продолжить обучение - потери высоки")
                elif avg_loss > 1.0:
                    report.append("• Модель показывает удовлетворительные результаты")
                else:
                    report.append("• Модель показывает хорошие результаты")
                    
                # Анализ градиентов
                grad_norm = weights_info.get('total_gradient_norm', 0) if weights_info else 0
                if grad_norm < 0.01:
                    report.append("• Градиенты очень малы - возможно, модель застряла в локальном минимуме")
                elif grad_norm > 10.0:
                    report.append("• Градиенты велики - возможно, нужна регуляризация или уменьшение learning rate")
            else:
                report.append("• Нет данных для анализа - требуется обучение")
                
            report.append("=" * 50)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {e}")
            logger.error(traceback.format_exc())
            return f"Ошибка при генерации отчета: {e}"

def main():
    """Основная функция для демонстрации работы"""
    # Создание чат-бота
    bot = SinChatBot(model_path=MODEL_DIR)
    
    print("Добро пожаловать в чат-бот Sin!")
    print("Доступные команды:")
    print("1. /train - обучение на данных")
    print("2. /dialogue - обучение на диалоге")
    print("3. /chat - режим чата")
    print("4. /report - отчет о состоянии модели")
    print("5. /exit - выход")
    
    if not TRANSFORMERS_AVAILABLE:
        print("\nВНИМАНИЕ: Библиотека 'transformers' не найдена. Будет использован кастомный токенизатор.")
        print("Для лучшего качества рекомендуется установить её: pip install transformers")
    
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
                    
            elif command == '/report':
                report = bot.get_model_report()
                print(report)
                # Также сохраняем отчет в файл
                report_file = os.path.join(PROJECT_DIR, f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"Отчет также сохранен в файл: {report_file}")
                    
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
