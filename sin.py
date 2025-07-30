import torch
# Исправлен импорт autocast для PyTorch >= 2.4
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import re
import glob
import time
import math
from datetime import datetime
from collections import Counter, defaultdict, deque # deque для истории чата
import logging
from torch.nn.utils.rnn import pad_sequence
import json
import matplotlib.pyplot as plt
import psutil  # Для мониторинга системных ресурсов
import gc     # Для ручной очистки памяти
# Добавлены импорты для парсинга веб-страниц
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
# Установите psutil: pip install psutil
# Для работы с DOCX файлами
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️  python-docx не установлен. Установите его для поддержки .docx файлов: pip install python-docx")
# Для работы с PDF файлами
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyPDF2 не установлен. Установите его для поддержки .pdf файлов: pip install PyPDF2")
# Для реальной BPE токенизации
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_SUPPORT = True
except ImportError:
    TOKENIZERS_SUPPORT = False
    print("⚠️  tokenizers не установлен. Установите его для поддержки BPE: pip install tokenizers")
# --- Импорты для сжатия ---
try:
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    print("⚠️  sklearn не установлен. Установите его для поддержки сжатия моделей: pip install scikit-learn")
import heapq
import struct
import io
# --------------------------

# --- Новый базовый путь ---
BASE_DIR = r"C:\Users\User\Downloads"
# --------------------------
# Создание директории для моделей
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
# --- Постоянный токенайзер ---
PERSISTENT_TOKENIZER_PATH = os.path.join(MODELS_DIR, "persistent_tokenizer.json")
# --------------------------
# Создание необходимых директорий
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
# ------------------
# Настройка логирования
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'ai_log.txt'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ------------------
# Гиперпараметры (уменьшены для CPU)
# ------------------
DEFAULT_SEQ_LENGTH = 256   # Уменьшено с 512
DEFAULT_BATCH_SIZE = 8     # Уменьшено с 16
DEFAULT_EPOCHS = 50        # Уменьшено с 100
DEFAULT_LEARNING_RATE = 3e-4
# --- Измененные параметры ---
DEFAULT_HIDDEN_SIZE = 256  # Увеличено с 128
DEFAULT_NUM_LAYERS = 8     # Увеличено с 2
# ---------------------------
DEFAULT_ATTENTION_HEADS = 8 # Уменьшено с 12
DEFAULT_FF_HIDDEN_SIZE = 2048 # Уменьшено с 3072
DEFAULT_DROPOUT = 0.1
MAX_SAVED_MODELS = 5
DEFAULT_TOKEN_TYPE = "bpe" # Теперь это будет означать использование tokenizers BPE
DEFAULT_MODEL_TYPE = "gpt"
DEFAULT_COMPRESSION_N_CLUSTERS = 256 # По умолчанию для VQ
# Имя ассистента
ASSISTANT_NAME = "Sin"
# ------------------
# Адаптивные конфигурации (обновлены значения по умолчанию для CPU)
# ------------------
ADAPTIVE_CONFIGS = {
    "high_end_gpu": {
        "seq_length": 512,
        "batch_size": 16,
        "epochs": 100,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "ff_hidden_size": 3072,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "token_type": "bpe"
    },
    "mid_end_gpu": {
        "seq_length": 256,
        "batch_size": 8,
        "epochs": 75,
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 8,
        "ff_hidden_size": 2048,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "token_type": "bpe"
    },
    "low_end_gpu": {
        "seq_length": 128,
        "batch_size": 4,
        "epochs": 50,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "ff_hidden_size": 1024,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "token_type": "bpe"
    },
    "high_memory_cpu": {
        "seq_length": 256,
        "batch_size": 4,
        "epochs": 30,
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "ff_hidden_size": 2048,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "token_type": "bpe"
    },
    "mid_memory_cpu": {
        "seq_length": 128,
        "batch_size": 2,
        "epochs": 20,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "ff_hidden_size": 1024,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "token_type": "bpe"
    },
    # --- Обновленная конфигурация для CPU с увеличенными параметрами ---
    "low_memory_cpu": {
        "seq_length": 64,
        "batch_size": 1, # Остается 1
        "epochs": 10,
        "hidden_size": 256, # Увеличено до 256
        "num_layers": 8,    # Увеличено до 8
        "num_heads": 4,
        "ff_hidden_size": 1024,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "token_type": "bpe"
    }
    # ------------------------------------------------------------
}
# ------------------
# Layer Normalization
# ------------------
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
# ------------------
# Rotary Positional Embedding (RoPE)
# ------------------
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    def forward(self, positions):
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        position_ids_expanded = positions[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin):
    # Адаптация размерностей для правильного применения
    cos = cos.unsqueeze(1) # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1) # [batch_size, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
# ------------------
# Multi-Head Self-Attention
# ------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_length, _ = x.shape
        # Project to query, key, value
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply rotary positional embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply attention mask
        if attention_mask is not None:
            # Ensure correct shape for masking
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        # Output projection
        output = self.o_proj(attn_output)
        return output, attn_weights
# ------------------
# Feed-Forward Network
# ------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ff_hidden_size)
        self.linear2 = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
# ------------------
# Transformer Block
# ------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, ff_hidden_size, dropout)
        self.ln1 = LayerNorm(hidden_size)
        self.ln2 = LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, attention_mask=None, position_ids=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(self.ln1(x), attention_mask, position_ids)
        x = x + self.dropout1(attn_output)
        # Feed-forward with residual connection
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_output)
        return x, attn_weights
# ------------------
# Современная GPT-Style модель
# ------------------
class ModernGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8,
                 ff_hidden_size=2048, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        # Token embeddings (без позиционных, используем RoPE)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        # Final layer normalization
        self.ln_f = LayerNorm(hidden_size)
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        # Initialize weights
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_length = input_ids.shape
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        # Embeddings (only token embeddings as we use RoPE)
        x = self.token_embedding(input_ids)
        # Apply transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask, position_ids)
            attention_weights.append(attn_weights)
        # Final layer normalization
        x = self.ln_f(x)
        # Language modeling head
        logits = self.lm_head(x)
        return logits, attention_weights
    def generate(self, input_ids, max_new_tokens, eos_token_id, temperature=1.0, do_sample=True):
        """Генерация с ранней остановкой по EOS токену."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                # Ранняя остановка
                if next_token.item() == eos_token_id:
                    break
        return input_ids
    def get_model_info(self):
        info = f"ModernGPT Model:\n"
        info += f"  Vocabulary size: {self.token_embedding.num_embeddings}\n"
        info += f"  Hidden size: {self.hidden_size}\n"
        info += f"  Number of layers: {self.num_layers}\n"
        info += f"  Attention heads: {self.blocks[0].attention.num_heads}\n"
        info += f"  Feed-forward hidden size: {self.blocks[0].ffn.linear1.out_features}\n"
        info += f"  Max sequence length: {self.max_seq_length}\n"
        info += f"  Parameters: {sum(p.numel() for p in self.parameters()):,}\n"
        info += f"  Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        return info
# ------------------
# IterableDataset для потоковой обработки (улучшенная реализация worker split)
# ------------------
class StreamingTextIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, tokenizer, seq_length, stride=None, chunk_size=1024*1024): # 1MB chunks
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length // 2
        self.chunk_size = chunk_size
        # Добавим оценку длины для совместимости (не точная)
        self._estimated_length = self._estimate_length()
    def _estimate_length(self):
        """Оценка количества последовательностей в файле."""
        try:
            file_size = os.path.getsize(self.file_path)
            # Это очень грубая оценка. В реальности зависит от токенизатора.
            # Предположим в среднем 4 символа на токен.
            estimated_tokens = file_size // 4
            if estimated_tokens >= self.seq_length:
                return (estimated_tokens - self.seq_length) // self.stride + 1
            else:
                return 0
        except OSError:
            return 0
    def __len__(self):
        # Возвращаем оценку, но помним, что она может быть неточной.
        # Это позволяет использовать len(dataset) в некоторых случаях, но с осторожностью.
        return self._estimated_length
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_handle = None
        file_size = os.path.getsize(self.file_path)
        start_offset = 0
        end_offset = file_size
        if worker_info is None:  # single-process loading
            logger.info("StreamingTextIterableDataset: Single worker mode.")
        else:  # in a worker process
            # Разделение данных между воркерами (если используется num_workers > 0)
            # Это усложняет логику, но обеспечивает паралелизм.
            # Делим файл на равные части по количеству воркеров.
            # Это может привести к разрыву последовательностей на границах,
            # но это приемлемый компромисс для потоковой обработки.
            logger.info(f"StreamingTextIterableDataset: Worker {worker_info.id} of {worker_info.num_workers}")
            per_worker = int(math.ceil(file_size / float(worker_info.num_workers)))
            start_offset = worker_info.id * per_worker
            end_offset = min(start_offset + per_worker, file_size)
            logger.info(f"Worker {worker_info.id} will process bytes {start_offset} to {end_offset} (size: {end_offset - start_offset})")
        try:
            # Открываем файл и устанавливаем начальную позицию
            file_handle = open(self.file_path, 'r', encoding='utf-8', errors='ignore')
            file_handle.seek(start_offset)
            # Если это не первый воркер, нам нужно найти начало следующего "полного" чанка/предложения/строки
            # чтобы избежать разрывов внутри слов/токенов. Простейший способ - пропустить до конца текущей строки.
            if worker_info is not None and worker_info.id > 0:
                # Пропускаем остаток строки, чтобы начать с новой
                file_handle.readline()
                logger.debug(f"Worker {worker_info.id} skipped to start of next line.")
            buffer_tokens = []
            bytes_read = start_offset
            # Если это не первый воркер, начальный индекс токенов в буфере может быть не 0
            # из-за пропущенной строки. Но для простоты логики генерации последовательностей
            # мы будем считать, что генерация начинается с начала буфера.
            # Это может привести к небольшому дублированию или пропуску последовательностей на границах,
            # но в большинстве случаев это не критично для потокового обучения.
            while bytes_read < end_offset:
                # Читаем чанк, но не больше, чем осталось до границы воркера
                read_size = min(self.chunk_size, end_offset - bytes_read)
                chunk = file_handle.read(read_size)
                if not chunk:
                    break
                bytes_read += len(chunk.encode('utf-8', errors='ignore')) # Приблизительный подсчет байт
                # Токенизируем чанк
                chunk_tokens = self.tokenizer.encode(chunk).ids
                buffer_tokens.extend(chunk_tokens)
                # Генерируем последовательности из буфера
                i = 0
                while i + self.seq_length + 1 <= len(buffer_tokens):
                    x = buffer_tokens[i:i+self.seq_length]
                    y = buffer_tokens[i+1:i+self.seq_length+1]
                    yield (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
                    i += self.stride
                # Оставляем в буфере только неполные последовательности для следующего чанка
                # Это важно для корректной обработки перекрывающихся последовательностей
                if len(buffer_tokens) > self.seq_length:
                    # Оставляем последние токены, которые могут быть началом новой последовательности
                    # Используем более точную логику перекрытия
                    overlap_start_index = len(buffer_tokens) - ((len(buffer_tokens) - self.seq_length - 1) % self.stride + self.seq_length + 1)
                    if overlap_start_index < 0: overlap_start_index = 0
                    buffer_tokens = buffer_tokens[overlap_start_index:]
        finally:
            if file_handle:
                file_handle.close()
# ------------------
# Класс для сбора метрик
# ------------------
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'training_loss': [],
            'validation_loss': [],
            'training_perplexity': [],
            'validation_perplexity': [],
            'learning_rate': [],
            'gradient_norm': [],
            'epoch_times': [],
            'batch_losses': [],
            'weight_statistics': {},
            'grad_norm_by_layer': {},
            'attention_weights_stats': {},
            'system_resources': []
        }
    def add_training_loss(self, loss):
        self.metrics['training_loss'].append(loss)
    def add_validation_loss(self, loss):
        self.metrics['validation_loss'].append(loss)
    def add_training_perplexity(self, perplexity):
        self.metrics['training_perplexity'].append(perplexity)
    def add_validation_perplexity(self, perplexity):
        self.metrics['validation_perplexity'].append(perplexity)
    def add_learning_rate(self, lr):
        self.metrics['learning_rate'].append(lr)
    def add_gradient_norm(self, grad_norm):
        self.metrics['gradient_norm'].append(grad_norm)
    def add_epoch_time(self, time_taken):
        self.metrics['epoch_times'].append(time_taken)
    def add_batch_loss(self, loss):
        self.metrics['batch_losses'].append(loss)
    def add_system_resources(self):
        # Сбор информации о системных ресурсах
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        self.metrics['system_resources'].append({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3)
        })
    def collect_weight_statistics(self, model):
        weight_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'shape': list(param.data.shape)
                }
        self.metrics['weight_statistics'] = weight_stats
        return weight_stats
    def collect_gradient_norms(self, model):
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        self.metrics['grad_norm_by_layer'] = grad_norms
        return grad_norms
    def collect_attention_stats(self, attention_weights):
        if attention_weights and len(attention_weights) > 0:
            last_layer_attn = attention_weights[-1]
            if len(last_layer_attn.shape) >= 2:
                attn_stats = {
                    'mean': last_layer_attn.mean().item(),
                    'std': last_layer_attn.std().item(),
                    'max': last_layer_attn.max().item(),
                    'min': last_layer_attn.min().item(),
                }
                self.metrics['attention_weights_stats'] = attn_stats
    def save_metrics(self, filename):
        filepath = os.path.join(METRICS_DIR, filename)
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value]
            elif isinstance(value, dict):
                serializable_metrics[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        serializable_metrics[key][subkey] = {}
                        for subsubkey, subsubvalue in subvalue.items():
                            if isinstance(subsubvalue, np.integer):
                                serializable_metrics[key][subkey][subsubkey] = int(subsubvalue)
                            elif isinstance(subsubvalue, np.floating):
                                serializable_metrics[key][subkey][subsubkey] = float(subsubvalue)
                            elif isinstance(subsubvalue, np.ndarray):
                                serializable_metrics[key][subkey][subsubkey] = subsubvalue.tolist()
                            else:
                                serializable_metrics[key][subkey][subsubkey] = subsubvalue
                    else:
                        serializable_metrics[key][subkey] = subvalue
            else:
                serializable_metrics[key] = value
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Метрики сохранены в {filepath}")
    def plot_metrics(self, filename_prefix):
        try:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            if self.metrics['training_loss']:
                plt.plot(self.metrics['training_loss'], label='Training Loss')
            if self.metrics['validation_loss']:
                plt.plot(self.metrics['validation_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 3, 2)
            if self.metrics['training_perplexity']:
                plt.plot(self.metrics['training_perplexity'], label='Training Perplexity')
            if self.metrics['validation_perplexity']:
                plt.plot(self.metrics['validation_perplexity'], label='Validation Perplexity')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')
            plt.title('Training and Validation Perplexity')
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 3, 3)
            if self.metrics['learning_rate']:
                plt.plot(self.metrics['learning_rate'], label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 3, 4)
            if self.metrics['gradient_norm']:
                plt.plot(self.metrics['gradient_norm'], label='Gradient Norm')
            plt.xlabel('Epoch')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norm')
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 3, 5)
            if self.metrics['epoch_times']:
                plt.plot(self.metrics['epoch_times'], label='Epoch Time')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.title('Training Time per Epoch')
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 3, 6)
            if self.metrics['batch_losses'] and len(self.metrics['batch_losses']) > 10:
                recent_losses = self.metrics['batch_losses'][-100:]
                plt.plot(recent_losses, label='Recent Batch Losses')
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                plt.title('Recent Batch Losses')
                plt.legend()
                plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(METRICS_DIR, f"{filename_prefix}_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Графики метрик сохранены в {plot_path}")
        except Exception as e:
            logger.error(f"Ошибка при построении графиков: {e}")
# ------------------
# Продвинутые техники сэмплирования
# ------------------
def advanced_sampling(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, previous_tokens=None):
    if repetition_penalty != 1.0 and previous_tokens is not None:
        for token_id in set(previous_tokens):
            logits[:, token_id] /= repetition_penalty
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(sorted_indices.size(0)):
            indices_to_remove[i, sorted_indices[i, sorted_indices_to_remove[i]]] = True
        logits[indices_to_remove] = float('-inf')
    return F.softmax(logits, dim=-1)
# ------------------
# Early Stopping
# ------------------
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
# ------------------
# Label Smoothing Loss
# ------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
# ------------------
# Gradient Noise
# ------------------
def add_gradient_noise(optimizer, sigma=1e-3):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * sigma
                param.grad.add_(noise)
# ------------------
# Подготовка данных с tokenizers (обновлено)
# ------------------
def train_tokenizer(files, vocab_size=30000, special_tokens=None):
    if not TOKENIZERS_SUPPORT:
        raise Exception("Поддержка tokenizers не доступна. Установите tokenizers")
    # Добавлены специальные токены для диалога
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<USER>', '<BOT>']
    tokenizer = Tokenizer(BPE(unk_token='<UNK>'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
    tokenizer.train(files, trainer)
    # Установка пост-процессора для автоматического добавления BOS/EOS
    # Убран автоматический BOS/EOS, так как они будут добавляться вручную или модель будет учится без них.
    # tokenizer.post_processor = TemplateProcessing(
    #     single="<BOS> $A <EOS>",
    #     special_tokens=[("<BOS>", special_tokens.index("<BOS>")), ("<EOS>", special_tokens.index("<EOS>"))],
    # )
    return tokenizer
def tokenize_with_tokenizer(tokenizer, text):
    if not TOKENIZERS_SUPPORT:
        raise Exception("Поддержка tokenizers не доступна. Установите tokenizers")
    encoding = tokenizer.encode(text)
    return encoding.ids
def detokenize_with_tokenizer(tokenizer, ids):
    if not TOKENIZERS_SUPPORT:
        raise Exception("Поддержка tokenizers не доступна. Установите tokenizers")
    return tokenizer.decode(ids)
# ------------------
# Новая функция для загрузки текста с URL
# ------------------
def load_text_from_url(url):
    """
    Загружает текст с веб-страницы по URL.
    Пытается извлечь основной текстовой контент.
    """
    logger.info(f"Попытка загрузки текста с URL: {url}")
    try:
        # Проверка URL формально (не обязательно, но полезно)
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Недопустимый формат URL")
        # Выполнение HTTP-запроса
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        } # Некоторые сайты блокируют запросы без User-Agent
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status() # Проверка на ошибки HTTP (4xx, 5xx)
        # Проверка типа контента
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            logger.warning(f"URL {url} не является HTML-страницей (Content-Type: {content_type}). Пробуем загрузить как текст.")
            # Если это не HTML, попробуем загрузить как текст
            try:
                # Попробуем декодировать как текст
                text_content = response.text
                if text_content:
                    logger.info(f"Текст успешно загружен с {url} как не-HTML контент. Длина: {len(text_content)} символов.")
                    return text_content
                else:
                    logger.warning(f"Не удалось извлечь текст с {url} как не-HTML контент.")
                    return ""
            except Exception as decode_e:
                logger.error(f"Ошибка декодирования не-HTML контента с {url}: {decode_e}")
                raise Exception(f"Ошибка декодирования контента с {url}: {decode_e}")
        # Парсинг HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # --- Стратегии извлечения текста ---
        # 1. Попробовать найти основной контент по типичным тегам/классам
        #    Это потребует адаптации под типичные сайты, которые вы хотите парсить.
        #    Примеры (нужно адаптировать под конкретные сайты):
        content_selectors = [
            'article',
            '[class*="content"]', # Атрибут class содержит "content"
            '[class*="article"]',
            '.post-body',
            '.entry-content',
            'main',
            'div.content',
            '.post-content',
            '.article-body',
            '#content',
            '.main-content'
        ]
        text_content = ""
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                # Удаление скриптов, стилей, навигации и т.д. из найденного блока
                for script in content(["script", "style", "nav", "aside", "footer", "header"]):
                    script.decompose()
                text_content = content.get_text(separator=' ', strip=True)
                if len(text_content) > 100: # Минимальная длина для "реального" контента
                     logger.info(f"Текст извлечен с использованием селектора: {selector}")
                     break
                else:
                     text_content = "" # Слишком короткий, пробуем следующий селектор
        # 2. Если специфические селекторы не сработали, попробовать более общий подход
        if not text_content:
            logger.info("Специфические селекторы не сработали, пробуем общий подход.")
            # Удаление потенциально ненужных тегов со всей страницы
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
                tag.decompose()
            # Извлечение текста из <body> или всего документа
            body = soup.find('body')
            if body:
                text_content = body.get_text(separator=' ', strip=True)
            else:
                text_content = soup.get_text(separator=' ', strip=True)
        if text_content:
            logger.info(f"Текст успешно загружен с {url}. Длина: {len(text_content)} символов.")
            return text_content
        else:
            logger.warning(f"Не удалось извлечь текст с {url}")
            return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка сети при запросе {url}: {e}")
        raise Exception(f"Ошибка при загрузке URL {url}: {e}")
    except Exception as e:
        logger.error(f"Ошибка при парсинге {url}: {e}")
        raise Exception(f"Ошибка при обработке содержимого URL {url}: {e}")
# ------------------
# Новые функции для загрузки и обработки JSON
# ------------------
def load_json_file(file_path):
    """Загружает данные из JSON файла."""
    logger.info(f"Попытка загрузки JSON файла: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON файл {file_path} успешно загружен. Количество записей: {len(data) if isinstance(data, list) else 'N/A'}")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке JSON файла {file_path}: {e}")
        raise
def process_json_to_dialogue_text(json_data):
    """
    Преобразует данные JSON в форматированный текст диалога.
    Поддерживает два формата:
    1. [{"instruction": "...", "input": "...", "output": "..."}, ...]
    2. [{"input": "...", "output": "..."}, ...]
    """
    logger.info("Начало преобразования JSON в текст диалога...")
    dialogue_texts = []
    if not isinstance(json_data, list):
        logger.warning("JSON данные не являются списком. Попытка обработать как один элемент.")
        json_data = [json_data]
    for item in json_data:
        try:
            # Формат 1: instruction + input + output
            if "instruction" in item and "input" in item and "output" in item:
                instruction = item.get("instruction", "").strip()
                user_input = item.get("input", "").strip()
                bot_output = item.get("output", "").strip()
                if bot_output: # Только если есть ответ
                    # Формируем контекст: инструкция + вход
                    context_parts = []
                    if instruction:
                        context_parts.append(instruction)
                    if user_input:
                        context_parts.append(user_input)
                    context = " ".join(context_parts)
                    dialogue_text = f"<USER>{context}<EOS><BOT>{bot_output}<EOS>"
                    dialogue_texts.append(dialogue_text)
            # Формат 2: input + output
            elif "input" in item and "output" in item:
                user_input = item.get("input", "").strip()
                bot_output = item.get("output", "").strip()
                if user_input and bot_output: # Только если есть и запрос, и ответ
                    dialogue_text = f"<USER>{user_input}<EOS><BOT>{bot_output}<EOS>"
                    dialogue_texts.append(dialogue_text)
            else:
                logger.warning(f"Пропущена запись JSON с неожиданным форматом: {item.keys()}")
        except Exception as e:
            logger.warning(f"Ошибка при обработке записи JSON {item}: {e}")
            continue
    combined_text = "\n".join(dialogue_texts)
    logger.info(f"Преобразование JSON завершено. Обработано {len(dialogue_texts)} диалогов. Общий размер текста: {len(combined_text)} символов.")
    return combined_text
# ------------------
# Обновленные функции загрузки текста
# ------------------
def load_text(file_path):
    """Универсальная функция загрузки текста из различных форматов файлов."""
    logger.info(f"Попытка загрузки файла: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.json':
            # Загрузка и обработка JSON
            json_data = load_json_file(file_path)
            text = process_json_to_dialogue_text(json_data)
            return text
        elif file_extension == '.txt':
            return load_txt_file(file_path)
        elif file_extension == '.docx':
            return load_docx_file(file_path)
        elif file_extension == '.pdf':
            return load_pdf_file(file_path)
        else:
            logger.warning(f"Неизвестный формат файла {file_extension}, пробуем загрузить как текст")
            return load_txt_file(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
        raise
def load_txt_file(file_path):
    encodings = ['utf-8', 'windows-1251', 'cp1251', 'koi8-r', 'latin1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            logger.info(f"Файл {file_path} успешно загружен с кодировкой {encoding}")
            return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path} с кодировкой {encoding}: {e}")
            continue
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logger.warning(f"Файл {file_path} загружен с игнорированием ошибок кодировки")
        return text
    except Exception as e:
        raise Exception(f"Не удалось загрузить файл {file_path} ни с одной кодировкой: {e}")
def load_docx_file(file_path):
    if not DOCX_SUPPORT:
        raise Exception("Поддержка DOCX файлов не доступна. Установите python-docx")
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        logger.info(f"DOCX файл {file_path} успешно загружен")
        return text
    except Exception as e:
        raise Exception(f"Ошибка при загрузке DOCX файла {file_path}: {e}")
def load_pdf_file(file_path):
    if not PDF_SUPPORT:
        raise Exception("Поддержка PDF файлов не доступна. Установите PyPDF2")
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        logger.info(f"PDF файл {file_path} успешно загружен")
        return text
    except Exception as e:
        raise Exception(f"Ошибка при загрузке PDF файла {file_path}: {e}")
def clean_text(text):
    original_length = len(text)
    # Исправленная строка с корректным экранированием апострофа
    # Оставляем больше специальных символов для диалогов и JSON
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\"\\\'\u0400-\u04FF<>/\[\]{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    cleaned_length = len(text)
    logger.info(f"Текст очищен: {original_length} -> {cleaned_length} символов")
    return text.strip()
# ------------------
# Управление моделями
# ------------------
def get_model_files():
    # Теперь ищем только сжатые модели
    model_files = glob.glob(os.path.join(MODELS_DIR, "compressed_gpt_model_*.pth"))
    model_files.sort(key=os.path.getctime, reverse=True)
    return model_files

def cleanup_old_models():
    model_files = get_model_files()
    if len(model_files) > MAX_SAVED_MODELS:
        old_models = model_files[MAX_SAVED_MODELS:]
        for old_model in old_models:
            try:
                os.remove(old_model)
                logger.info(f"Удалена старая модель: {os.path.basename(old_model)}")
            except Exception as e:
                logger.error(f"Ошибка при удалении модели {old_model}: {e}")

# Исправленная функция загрузки модели с weights_only=False
def save_compressed_model_with_timestamp(model, tokenizer_path, vocab_size,
                            loss=0.0, token_type="bpe", perplexity=None,
                            training_config=None, metrics_collector=None, model_type="gpt",
                            n_clusters=DEFAULT_COMPRESSION_N_CLUSTERS):
    """Сохраняет только сжатую модель."""
    if not KMEANS_AVAILABLE:
        logger.warning("sklearn не установлен. Сохранение обычной модели.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"gpt_model_{timestamp}.pth" # Обычное имя, если сжатие невозможно
        model_path = os.path.join(MODELS_DIR, model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_path': tokenizer_path,
            'vocab_size': vocab_size,
            'timestamp': timestamp,
            'loss': loss,
            'perplexity': perplexity,
            'token_type': token_type,
            'model_type': model_type,
            'training_config': training_config,
            'model_config': {
                'hidden_size': getattr(model, 'hidden_size', 512),
                'num_layers': getattr(model, 'num_layers', 6),
                'num_heads': getattr(model, 'blocks', [None])[0].attention.num_heads if hasattr(model, 'blocks') and len(model.blocks) > 0 else 8,
                'ff_hidden_size': getattr(model, 'blocks', [None])[0].ffn.linear1.out_features if hasattr(model, 'blocks') and len(model.blocks) > 0 else 2048,
                'dropout': getattr(model, 'blocks', [None])[0].dropout1.p if hasattr(model, 'blocks') and len(model.blocks) > 0 else 0.1,
                'model_type': type(model).__name__
            }
        }, model_path)
        logger.info(f"Модель (без сжатия) сохранена: {model_path}")
        if metrics_collector:
            metrics_filename = f"metrics_{timestamp}.json"
            metrics_collector.save_metrics(metrics_filename)
            metrics_collector.plot_metrics(f"metrics_{timestamp}")
        cleanup_old_models()
        return model_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"compressed_gpt_model_{timestamp}.pth" # Имя для сжатой модели
    model_path = os.path.join(MODELS_DIR, model_filename)
    try:
        # --- Сжатие ---
        logger.info(f"Начало сжатия модели в {model_path}...")
        compressed_data = {}
        total_original_size = 0
        total_compressed_size = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Сжатие слоя: {name}")
                weights_np = param.data.cpu().numpy()
                original_size = weights_np.nbytes
                total_original_size += original_size

                # 1. Векторное квантование
                indices, centers, shape = quantize_layer(weights_np, n_clusters=n_clusters)
                
                # 2. Простое сохранение индексов и центров
                compressed_data[name] = {
                    'indices': indices,
                    'centers': centers,
                    'shape': shape
                }
                
                # Оценка размера (приблизительная)
                compressed_size = indices.nbytes + centers.nbytes
                total_compressed_size += compressed_size
                logger.debug(f"  Оригинал: {original_size} байт, Сжато: ~{compressed_size} байт")

        # Сохраняем в файл
        torch.save(compressed_data, model_path)
        compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else float('inf')
        logger.info(f"Сжатие модели завершено. Соотношение: {compression_ratio:.2f}x. Файл: {model_path}")
        print(f"✅ Модель успешно сжата и сохранена в {os.path.basename(model_path)} (Соотношение: {compression_ratio:.2f}x)")
        # --- Конец сжатия ---
        
        # Сохраняем метрики и т.д. (это можно улучшить, сохранив их отдельно или в том же файле)
        if metrics_collector:
            metrics_filename = f"metrics_{timestamp}.json"
            metrics_collector.save_metrics(metrics_filename)
            metrics_collector.plot_metrics(f"metrics_{timestamp}")
        cleanup_old_models()
        return model_path
    except Exception as e:
        logger.error(f"Ошибка при сохранении сжатой модели: {e}")
        return None

def load_compressed_model(model, compressed_model_path, device="cpu"):
    """Загружает сжатую модель и восстанавливает веса в переданную модель."""
    try:
        logger.info(f"Начало загрузки сжатой модели из {compressed_model_path}...")
        compressed_data = torch.load(compressed_model_path, weights_only=False) # weights_only=False для загрузки словаря
        
        for name, param in model.named_parameters():
            if name in compressed_data and param.requires_grad:
                layer_data = compressed_data[name]
                
                # Восстановление индексов и центров
                indices = layer_data['indices']
                centers = layer_data['centers']
                shape = layer_data['shape']
                
                # Восстановление весов из индексов и центров
                weights = centers[indices].reshape(shape)
                
                # Загрузка весов в модель
                param.data = torch.tensor(weights, dtype=param.dtype, device=device)
                
        logger.info("Сжатая модель успешно загружена и веса восстановлены.")
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке сжатой модели: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_model_with_dicts(model_path, device):
    """
    Загружает модель из сжатого файла.
    model_path - путь к сжатому файлу.
    """
    try:
        # Установка weights_only=False для совместимости с PyTorch 2.6+
        # Загружаем чекпоинт, чтобы получить конфигурацию модели
        # Предполагаем, что чекпоинт содержит необходимую информацию
        # В реальном сценарии, эту информацию нужно сохранять в сжатый файл тоже.
        # Здесь мы попробуем получить её из имени файла или из отдельного файла конфигурации.
        # Для простоты, предположим, что мы можем извлечь её из оригинального имени файла
        # или что она была сохранена отдельно.
        # В этом примере, мы будем использовать информацию из training_config, если она доступна.
        # Но для загрузки нам нужно знать архитектуру. Попробуем извлечь её из имени файла.
        # Это не самый надежный способ, но сработает для нашего случая.
        
        # Загружаем данные из сжатого файла для получения конфигурации
        compressed_data = torch.load(model_path, map_location='cpu', weights_only=False)
        # Предполагаем, что в сжатом файле есть ключ 'training_config' или 'model_config'
        # В реальной реализации это должно быть частью сохраненного состояния.
        # Здесь мы делаем предположение, что чекпоинт содержит эту информацию.
        # Это не идеально, но для демонстрации сойдет.
        # Лучше было бы сохранять конфигурацию модели в отдельный файл или в тот же сжатый файл.
        # Попробуем загрузить чекпоинт, чтобы получить конфигурацию.
        # Так как мы сохраняем только сжатую модель, нам нужно где-то хранить конфигурацию.
        # Можно сохранить её в отдельный .json файл с тем же именем.
        config_path = model_path.replace(".pth", ".json")
        model_config = {}
        training_config = {}
        if os.path.exists(config_path):
             try:
                 with open(config_path, 'r') as f:
                     config_data = json.load(f)
                 model_config = config_data.get('model_config', {})
                 training_config = config_data.get('training_config', {})
                 logger.info(f"Конфигурация загружена из {config_path}")
             except Exception as e:
                 logger.warning(f"Не удалось загрузить конфигурацию из {config_path}: {e}")
        else:
             logger.warning(f"Файл конфигурации {config_path} не найден. Используются значения по умолчанию.")
        
        vocab_size = training_config.get('vocab_size', model_config.get('vocab_size', 30000)) # fallback
        model_type = training_config.get('model_type', model_config.get('model_type', 'gpt'))
        
        # Создаем модель
        model = ModernGPT(
            vocab_size,
            model_config.get('hidden_size', 512),
            model_config.get('num_layers', 6),
            model_config.get('num_heads', 8),
            model_config.get('ff_hidden_size', 2048),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Загружаем веса из сжатого файла
        if load_compressed_model(model, model_path, device):
            logger.info(f"Модель загружена: {model_path}")
            # Возвращаем необходимые данные. Для совместимости с предыдущим кодом.
            tokenizer_path = training_config.get('tokenizer_path', PERSISTENT_TOKENIZER_PATH)
            timestamp = training_config.get('timestamp', 'unknown')
            loss = training_config.get('loss', 0.0)
            token_type = training_config.get('token_type', 'bpe')
            model_type = training_config.get('model_type', 'gpt')
            perplexity = training_config.get('perplexity', None)
            weight_stats = training_config.get('weight_statistics', {}) # Может быть пустым
            
            return model, tokenizer_path, token_type, model_type, perplexity, weight_stats, training_config
        else:
            logger.error("Не удалось загрузить веса из сжатого файла.")
            return None, None, None, None, None, None, None
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None, None

def list_available_models():
    model_files = get_model_files()
    if not model_files:
        print("Нет доступных моделей")
        return []
    print("\nДоступные сжатые модели:")
    for i, model_file in enumerate(model_files):
        try:
            # Попробуем загрузить конфигурацию из соседнего .json файла
            config_path = model_file.replace(".pth", ".json")
            config_data = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            
            timestamp = config_data.get('training_config', {}).get('timestamp', 'unknown')
            loss = config_data.get('training_config', {}).get('loss', 0.0)
            vocab_size = config_data.get('training_config', {}).get('vocab_size', 0)
            token_type = config_data.get('training_config', {}).get('token_type', 'bpe')
            model_type = config_data.get('training_config', {}).get('model_type', 'gpt')
            perplexity = config_data.get('training_config', {}).get('perplexity', 'N/A')
            
            print(f"{i+1}. {os.path.basename(model_file)}")
            print(f"   Дата: {timestamp}, Loss: {loss:.4f}, Perplexity: {perplexity}")
            print(f"   Vocab: {vocab_size}, Type: {token_type}, Model: {model_type}")
        except Exception as e:
            print(f"{i+1}. {os.path.basename(model_file)} (ошибка чтения: {e})")
    return model_files
# ------------------
# Определение профиля устройства
# ------------------
def detect_hardware_profile():
    """Определяет профиль устройства для адаптивного обучения."""
    profile = {
        "device": "cpu",
        "memory_gb": 0,
        "profile_name": "unknown"
    }
    if torch.cuda.is_available():
        profile["device"] = "cuda"
        props = torch.cuda.get_device_properties(0)
        profile["memory_gb"] = props.total_memory / (1024**3)
        # Примерная логика определения профиля
        if profile["memory_gb"] >= 10: # Например, 10 ГБ и больше
            profile["profile_name"] = "high_end_gpu"
        elif profile["memory_gb"] >= 6: # Например, от 6 до 10 ГБ
             profile["profile_name"] = "mid_end_gpu"
        else: # Меньше 6 ГБ
             profile["profile_name"] = "low_end_gpu" # Или "mid_end_gpu" с очень малым batch_size
    else:
        # Определение для CPU
        profile["device"] = "cpu"
        # Используем psutil для получения доступной памяти
        virtual_mem = psutil.virtual_memory()
        profile["memory_gb"] = virtual_mem.total / (1024**3)
        # Логика для CPU (очень приблизительная)
        if profile["memory_gb"] >= 16: # Например, 16 ГБ и больше
             profile["profile_name"] = "high_memory_cpu"
        elif profile["memory_gb"] >= 8: # Например, от 8 до 16 ГБ
             profile["profile_name"] = "mid_memory_cpu"
        else: # Меньше 8 ГБ
             profile["profile_name"] = "low_memory_cpu"
    return profile
# ------------------
# Расчет перплексии
# ------------------
def calculate_perplexity(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output, _ = model(x_batch)
            loss = criterion(output.reshape(-1, output.size(-1)), y_batch.reshape(-1))
            total_loss += loss.item() * x_batch.size(0) * x_batch.size(1)
            total_samples += x_batch.size(0) * x_batch.size(1)
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    return perplexity
# ------------------
# Генерация текста (обновлено)
# ------------------
def generate_text(model, tokenizer, start_tokens,
                 max_new_tokens=200, temperature=1.0, top_k=0, top_p=1.0,
                 repetition_penalty=1.0, device='cpu'):
    """Генерация текста с ранней остановкой и динамической длиной."""
    logger.info(f"Начало генерации текста: '{start_tokens}', max_new_tokens: {max_new_tokens}")
    logger.info(f"Параметры: температура={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}")
    model.eval()
    with torch.no_grad():
        # Токенизация начального текста
        input_ids_list = tokenize_with_tokenizer(tokenizer, start_tokens)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
        # Получаем ID токена EOS
        eos_token_id = tokenizer.token_to_id('<EOS>')
        if eos_token_id is None:
            logger.warning("Токен <EOS> не найден в токенайзере. Используется ID 0 как EOS.")
            eos_token_id = 0 # fallback
        # Генерация с ранней остановкой
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            do_sample=True # Всегда используем сэмплирование для генерации
        )
        # Декодируем только сгенерированную часть (без начального контекста)
        generated_ids = output_ids[0, len(input_ids_list):].tolist()
        # Удаляем EOS токен из финального текста, если он есть
        if generated_ids and generated_ids[-1] == eos_token_id:
            generated_ids = generated_ids[:-1]
        generated_text = detokenize_with_tokenizer(tokenizer, generated_ids)
        logger.info(f"Генерация завершена, сгенерировано {len(generated_ids)} токенов")
        return generated_text # Возвращаем только сгенерированный текст
# ------------------
# Улучшенный класс для управления историей чата (обновлено)
# ------------------
class ChatHistory:
    def __init__(self, tokenizer, max_context_tokens=512):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.history = deque() # Используем deque для эффективного добавления/удаления с обоих концов
        self.total_tokens = 0
    def add_user_message(self, message):
        entry = f"<USER>{message}<EOS>"
        self._add_entry(entry)
    def add_assistant_message(self, message):
        entry = f"<BOT>{message}<EOS>"
        self._add_entry(entry)
    def _add_entry(self, entry):
        tokens = tokenize_with_tokenizer(self.tokenizer, entry)
        self.history.append((entry, len(tokens)))
        self.total_tokens += len(tokens)
        # Обрезаем историю, если превышен лимит токенов
        while self.total_tokens > self.max_context_tokens and self.history:
            removed_entry, removed_tokens = self.history.popleft()
            self.total_tokens -= removed_tokens
    def get_context(self):
        # Собираем контекст из истории
        return "".join([entry for entry, _ in self.history])
    def clear(self):
        self.history.clear()
        self.total_tokens = 0
# ------------------
# Чат с ассистентом Sin (обновлено)
# ------------------
def chat_with_sin(model, tokenizer, device):
    if model is None or tokenizer is None:
        print("❌ Нет загруженной модели или токенайзера для чата.")
        return
    print(f"\n🗣️  Начинаем чат с {ASSISTANT_NAME}. Введите '/exit' для выхода или '/clear' для очистки истории.")
    # Используем улучшенный класс для управления историей
    chat_history = ChatHistory(tokenizer, max_context_tokens=384) # Оставляем запас
    model.eval()
    with torch.no_grad():
        while True:
            user_input = input("\nВы: ").strip()
            if user_input.lower() in ['/exit', '/quit']:
                print(f"{ASSISTANT_NAME}: До скорой встречи!")
                break
            elif user_input.lower() in ['/clear']:
                chat_history.clear()
                print(f"{ASSISTANT_NAME}: История диалога очищена.")
                continue
            # Формирование контекста с использованием улучшенного класса
            chat_history.add_user_message(user_input)
            context = chat_history.get_context() + "<BOT>"
            print(f"{ASSISTANT_NAME}: ", end='', flush=True)
            try:
                # Генерация ответа
                generated_text = generate_text(
                    model, tokenizer, context,
                    max_new_tokens=200, # Максимум, но генерация остановится на <EOS>
                    temperature=0.8, top_k=50, top_p=0.95,
                    repetition_penalty=1.1, device=device
                )
                # Вывод сгенерированного текста (без специальных токенов)
                # generated_text уже не содержит <BOT> в начале и <EOS> в конце благодаря generate_text
                print(generated_text.strip())
                # Обновление истории с ответом ассистента
                chat_history.add_assistant_message(generated_text.strip())
            except Exception as e:
                logger.error(f"Ошибка при генерации ответа: {e}")
                print(f"{ASSISTANT_NAME}: Извините, произошла ошибка при генерации ответа.")
# ------------------
# Модуль сжатия моделей (VQ)
# ------------------
def quantize_layer(weights, n_clusters=256):
    """Применяет векторное квантование к слою."""
    if not KMEANS_AVAILABLE:
        raise Exception("sklearn не установлен. Необходим для VQ.")
    shape = weights.shape
    flattened = weights.reshape(-1, 1)  # Преобразуем в 1D
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit(flattened)
    indices = kmeans.labels_.astype(np.uint8)  # 1 байт на индекс
    centers = kmeans.cluster_centers_.astype(np.float16)  # Центры в float16
    return indices, centers, shape
# ------------------
# Обучение с улучшенной обработкой ошибок (обновленная логика логирования)
# ------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device,
                tokenizer_path, vocab_size, token_type="bpe",
                learning_rate=DEFAULT_LEARNING_RATE, model_type="gpt",
                n_clusters_for_compression=DEFAULT_COMPRESSION_N_CLUSTERS):
    logger.info(f"Начало обучения модели на устройстве {device}")
    logger.info(f"Параметры обучения: epochs={epochs}, batch_size={train_loader.batch_size}")
    metrics_collector = MetricsCollector()
    training_config = {
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': learning_rate,
        'seq_length': train_loader.dataset.seq_length if hasattr(train_loader.dataset, 'seq_length') else 0,
        'vocab_size': vocab_size,
        'token_type': token_type,
        'model_type': model_type,
        'device': str(device),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'n_clusters_for_compression': n_clusters_for_compression,
        'tokenizer_path': tokenizer_path # Добавляем путь к токенайзеру
    }
    model.to(device)
    model.train()
    # Условное использование GradScaler для AMP
    use_amp = device.type == 'cuda' and torch.cuda.is_available()
    if use_amp:
        scaler = GradScaler()
        logger.info("AMP (Automatic Mixed Precision) включен для обучения.")
    else:
        # Используем DummyScaler для CPU
        class DummyScaler:
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = DummyScaler()
        logger.info("AMP отключен. Используется стандартная точность.")
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_loss = float('inf')
    best_perplexity = float('inf')
    best_model_path = None
    training_start_time = time.time()
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            total_batches = 0 # Будем считать динамически
            logger.info(f"Эпоха {epoch+1}/{epochs} начата")
            # Training phase
            model.train()
            # --- Обновленная логика логирования ---
            # Оцениваем общее количество батчей в эпохе (приблизительно)
            # Так как это IterableDataset, len(train_loader) не работает.
            # Мы можем оценить, если у dataset есть __len__, иначе используем -1.
            estimated_total_batches = len(train_loader.dataset) // train_loader.batch_size if hasattr(train_loader.dataset, '__len__') else -1
            log_interval = 100 # Логировать каждые 100 батчей
            # ---
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                try:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    # Используем autocast для AMP
                    with autocast(device_type=device.type, enabled=use_amp):
                        output, attention_weights = model(x_batch)
                        loss = criterion(output.reshape(-1, output.size(-1)), y_batch.reshape(-1))
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    add_gradient_noise(optimizer, sigma=1e-3)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    total_batches += 1
                    metrics_collector.add_batch_loss(loss.item())
                    # Логирование каждые log_interval батчей
                    # Формат: "Текущий_батч/Общее_батчей_в_эпохе" или "Текущий_батч/-" если общее неизвестно (например, для IterableDataset)
                    if batch_idx % log_interval == 0 and batch_idx > 0:
                        avg_batch_loss = total_loss / total_batches
                        progress_str = f"{batch_idx}/{estimated_total_batches}" if estimated_total_batches > 0 else f"{batch_idx}/-"
                        logger.info(f"Эпоха {epoch+1}/{epochs}, Батч {progress_str}, Loss: {avg_batch_loss:.4f}")
                        # Сбор метрик системы
                        metrics_collector.add_system_resources()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"Out of memory на батче {batch_idx}. Очистка памяти...")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        # Пропускаем этот батч и продолжаем
                        continue
                    else:
                        raise e
                except Exception as e:
                    logger.error(f"Ошибка в батче {batch_idx}: {e}")
                    continue # Продолжаем со следующего батча
            # Validation phase
            model.eval()
            val_loss = 0
            val_samples = 0
            val_batches = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    try:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        with autocast(device_type=device.type, enabled=use_amp): # AMP и для валидации
                            output, _ = model(x_batch)
                            loss = criterion(output.reshape(-1, output.size(-1)), y_batch.reshape(-1))
                        val_loss += loss.item() * x_batch.size(0) * x_batch.size(1)
                        val_samples += x_batch.size(0) * x_batch.size(1)
                        val_batches += 1
                    except Exception as e:
                        logger.error(f"Ошибка в валидационном батче: {e}")
                        continue
            if val_samples > 0:
                val_loss /= val_samples
            else:
                val_loss = float('inf')
            scheduler.step()
            # Calculate perplexity
            train_perplexity = np.exp(total_loss / max(total_batches, 1))
            val_perplexity = calculate_perplexity(model, val_loader, device, criterion)
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = total_loss / max(total_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            # Сбор метрик
            metrics_collector.add_training_loss(avg_train_loss)
            metrics_collector.add_validation_loss(val_loss)
            metrics_collector.add_training_perplexity(train_perplexity)
            metrics_collector.add_validation_perplexity(val_perplexity)
            metrics_collector.add_learning_rate(current_lr)
            metrics_collector.add_gradient_norm(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm))
            metrics_collector.add_epoch_time(epoch_time)
            metrics_collector.collect_gradient_norms(model)
            logger.info(f"Эпоха {epoch+1}/{epochs} завершена за {epoch_time:.2f} сек")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"  Train Perplexity: {train_perplexity:.4f}, Val Perplexity: {val_perplexity:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}, Gradient Norm: {grad_norm:.4f}")
            # Early stopping check
            if early_stopping(val_loss):
                logger.info(f"Early stopping на эпохе {epoch+1}")
                break
            # Сохранение лучшей модели (только сжатой)
            if val_loss < best_loss:
                best_loss = val_loss
                best_perplexity = val_perplexity
                metrics_collector.collect_weight_statistics(model)
                # --- Сохраняем только сжатую модель ---
                model_path = save_compressed_model_with_timestamp(model,
                                                     tokenizer_path,
                                                     vocab_size, val_loss, token_type, val_perplexity,
                                                     training_config, metrics_collector, model_type,
                                                     n_clusters=n_clusters_for_compression)
                # --- Сохраняем конфигурацию отдельно ---
                if model_path:
                    config_data_to_save = {
                        'model_config': {
                            'hidden_size': getattr(model, 'hidden_size', 512),
                            'num_layers': getattr(model, 'num_layers', 6),
                            'num_heads': getattr(model, 'blocks', [None])[0].attention.num_heads if hasattr(model, 'blocks') and len(model.blocks) > 0 else 8,
                            'ff_hidden_size': getattr(model, 'blocks', [None])[0].ffn.linear1.out_features if hasattr(model, 'blocks') and len(model.blocks) > 0 else 2048,
                            'dropout': getattr(model, 'blocks', [None])[0].dropout1.p if hasattr(model, 'blocks') and len(model.blocks) > 0 else 0.1,
                            'model_type': type(model).__name__,
                            'vocab_size': vocab_size
                        },
                        'training_config': training_config
                    }
                    config_path = model_path.replace(".pth", ".json")
                    try:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data_to_save, f, indent=2, ensure_ascii=False)
                        logger.info(f"Конфигурация модели сохранена в {config_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении конфигурации модели: {e}")
                # --------------------------------------
                if model_path:
                    best_model_path = model_path
                    logger.info(f"Новая лучшая модель (сжатая) сохранена: loss {val_loss:.4f}, perplexity {val_perplexity:.4f}")
            # Принудительная очистка памяти после каждой эпохи
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
    except KeyboardInterrupt:
        logger.info("Обучение прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка во время обучения: {e}")
        import traceback
        logger.error(traceback.format_exc())
    training_time = time.time() - training_start_time
    logger.info(f"Обучение завершено за {training_time:.2f} сек")
    logger.info(f"Лучшая модель: loss {best_loss:.4f}, perplexity {best_perplexity:.4f}")
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_collector.save_metrics(f"final_metrics_{final_timestamp}.json")
    metrics_collector.plot_metrics(f"final_metrics_{final_timestamp}")
    return best_model_path, metrics_collector
# ------------------
# Интерактивный режим
# ------------------
def interactive_mode():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Запуск интерактивного режима на устройстве: {device}")
    current_model = None
    current_tokenizer = None
    vocab_size = 0
    current_token_type = "bpe"
    current_model_type = "gpt"
    current_perplexity = None
    current_weight_stats = {}
    current_training_config = {}
    print("\n" + "="*80)
    print(f"🤖 Современный генеративный ИИ '{ASSISTANT_NAME}' с GPT-архитектурой (оптимизированная версия)")
    print("="*80)
    print(f"📁 Модели сохраняются в: {os.path.abspath(MODELS_DIR)}")
    print(f"📝 Логи сохраняются в: {os.path.abspath(LOGS_DIR)}")
    print(f"📊 Метрики сохраняются в: {os.path.abspath(METRICS_DIR)}")
    print(f"Кэширование в: {os.path.abspath(CACHE_DIR)}")
    print(f"Постоянный токенайзер: {os.path.abspath(PERSISTENT_TOKENIZER_PATH)}")
    print("\nДоступные команды:")
    print("  generate     - Генерация текста (продвинутая)")
    print("  train        - Обучение модели (поддерживаются файлы и URL)")
    print("  save         - Сохранение текущей модели (только сжатая)")
    print("  load         - Загрузка модели (только сжатая)")
    print("  list         - Список доступных моделей (только сжатые)")
    print("  info         - Информация о текущей модели")
    print("  weights      - Просмотр статистик весов")
    print("  metrics      - Просмотр метрик обучения")
    print("  chat         - Начать чат с ассистентом Sin")
    print("  quit         - Выход")
    print("="*80)
    # --- Автозагрузка последней модели ---
    try:
        model_files = get_model_files()
        if model_files:
            logger.info(f"Найдено {len(model_files)} сохраненных сжатых моделей. Попытка автозагрузки последней...")
            for i, latest_model_path in enumerate(model_files): # Цикл для повторных попыток
                model_basename = os.path.basename(latest_model_path)
                logger.info(f"Попытка загрузки модели {i+1}/{len(model_files)}: {model_basename}")
                try:
                    # Загружаем сжатую модель
                    loaded_model, loaded_tokenizer_path, loaded_token_type, loaded_model_type, loaded_perplexity, loaded_weight_stats, loaded_training_config = load_model_with_dicts(latest_model_path, device)
                    if loaded_model is not None:
                        current_model = loaded_model
                        current_token_type = loaded_token_type
                        current_model_type = loaded_model_type
                        current_perplexity = loaded_perplexity
                        current_weight_stats = loaded_weight_stats
                        current_training_config = loaded_training_config
                        vocab_size_from_checkpoint = loaded_training_config.get('vocab_size', 0)
                        # Попытка загрузить токенайзер
                        tokenizer_loaded_successfully = False
                        if os.path.exists(PERSISTENT_TOKENIZER_PATH) and TOKENIZERS_SUPPORT:
                            try:
                                current_tokenizer = Tokenizer.from_file(PERSISTENT_TOKENIZER_PATH)
                                vocab_size = current_tokenizer.get_vocab_size() if hasattr(current_tokenizer, 'get_vocab_size') else vocab_size_from_checkpoint
                                # Простая проверка работоспособности токенайзера
                                test_text = "проверка"
                                test_ids = tokenize_with_tokenizer(current_tokenizer, test_text)
                                test_decoded = detokenize_with_tokenizer(current_tokenizer, test_ids[:3]) # Декодируем часть для проверки
                                logger.info(f"✅ Постоянный токенайзер успешно загружен и протестирован.")
                                tokenizer_loaded_successfully = True
                            except Exception as tok_e:
                                logger.warning(f"⚠️ Ошибка при тестировании постоянного токенайзера из {PERSISTENT_TOKENIZER_PATH}: {tok_e}")
                                current_tokenizer = None
                        elif not os.path.exists(PERSISTENT_TOKENIZER_PATH):
                            logger.warning(f"⚠️ Постоянный файл токенайзера не найден: {PERSISTENT_TOKENIZER_PATH}")
                        elif not TOKENIZERS_SUPPORT:
                            logger.warning(f"⚠️ Библиотека tokenizers не доступна для загрузки токенайзера.")
                        if not tokenizer_loaded_successfully:
                            logger.warning(f"⚠️ Постоянный токенайзер не доступен или не работает. Будет использован vocab_size из чекпойнта.")
                            current_tokenizer = None
                            vocab_size = vocab_size_from_checkpoint
                        logger.info(f"✅ Модель успешно автозагружена из {model_basename}")
                        if current_perplexity is not None:
                            logger.info(f"   Perplexity модели: {current_perplexity:.2f}")
                        logger.info(f"   Vocabulary size: {vocab_size}")
                        print(f"✅ Автозагружена модель: {model_basename}" + (f" (Perplexity: {current_perplexity:.2f})" if current_perplexity is not None else ""))
                        break # Успешная загрузка, выходим из цикла попыток
                    else:
                        logger.warning(f"❌ Не удалось загрузить модель из {model_basename} (load_model_with_dicts вернул None)")
                except Exception as model_e:
                    logger.error(f"❌ Ошибка при загрузке модели {model_basename}: {model_e}")
                    if i == len(model_files) - 1: # Если это была последняя попытка
                        logger.error("Не удалось загрузить ни одну из доступных моделей.")
                    continue # Пробуем следующую модель в списке
        else:
            logger.info("Нет сохраненных сжатых моделей для автозагрузки.")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при попытке автозагрузки модели: {e}")
    # --- Конец автозагрузки ---
    while True:
        try:
            command = input("\nВведите команду: ").strip().lower()
            if command == "quit":
                if current_model is not None:
                    print("Автоматическое сохранение модели...")
                    # --- Сохраняем только сжатую модель ---
                    model_path = save_compressed_model_with_timestamp(current_model,
                                            PERSISTENT_TOKENIZER_PATH, # Используем постоянный путь
                                            vocab_size, token_type=current_token_type,
                                            model_type=current_model_type,
                                            perplexity=current_perplexity,
                                            training_config=current_training_config)
                    # --- Сохраняем конфигурацию отдельно ---
                    if model_path:
                        config_data_to_save = {
                            'model_config': {
                                'hidden_size': getattr(current_model, 'hidden_size', 512),
                                'num_layers': getattr(current_model, 'num_layers', 6),
                                'num_heads': getattr(current_model, 'blocks', [None])[0].attention.num_heads if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 8,
                                'ff_hidden_size': getattr(current_model, 'blocks', [None])[0].ffn.linear1.out_features if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 2048,
                                'dropout': getattr(current_model, 'blocks', [None])[0].dropout1.p if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 0.1,
                                'model_type': type(current_model).__name__,
                                'vocab_size': vocab_size
                            },
                            'training_config': current_training_config
                        }
                        config_path = model_path.replace(".pth", ".json")
                        try:
                            with open(config_path, 'w', encoding='utf-8') as f:
                                json.dump(config_data_to_save, f, indent=2, ensure_ascii=False)
                            logger.info(f"Конфигурация модели сохранена в {config_path}")
                        except Exception as e:
                            logger.error(f"Ошибка при сохранении конфигурации модели: {e}")
                    # --------------------------------------
                    if model_path:
                        print(f"✅ Модель (сжатая) сохранена в {os.path.basename(model_path)}")
                    else:
                        print("❌ Ошибка при сохранении модели")
                print("До свидания!")
                break
            elif command == "generate":
                if current_model is None or current_tokenizer is None:
                    print("❌ Нет загруженной модели или токенайзера. Сначала загрузите или обучите модель.")
                    continue
                start_text = input("Введите начальный текст: ").strip()
                if not start_text:
                    start_text = "машинное обучение"
                try:
                    temp = float(input("Температура (0.1-2.0, по умолчанию 1.0): ") or "1.0")
                    temp = max(0.1, min(2.0, temp))
                except ValueError:
                    temp = 1.0
                try:
                    top_k = int(input("Top-K (0 для отключения, по умолчанию 50): ") or "50")
                    top_k = max(0, top_k)
                except ValueError:
                    top_k = 50
                try:
                    top_p = float(input("Top-P (0.0-1.0, по умолчанию 0.9): ") or "0.9")
                    top_p = max(0.0, min(1.0, top_p))
                except ValueError:
                    top_p = 0.9
                try:
                    repetition_penalty = float(input("Repetition penalty (0.1-2.0, по умолчанию 1.0): ") or "1.0")
                    repetition_penalty = max(0.1, min(2.0, repetition_penalty))
                except ValueError:
                    repetition_penalty = 1.0
                try:
                    max_new_tokens = int(input("Максимум новых токенов (по умолчанию 200): ") or "200")
                    max_new_tokens = max(10, min(500, max_new_tokens))  # Уменьшено максимальное значение
                except ValueError:
                    max_new_tokens = 200
                print("\n🔄 Генерация текста...")
                try:
                    generated = generate_text(current_model, current_tokenizer,
                                            start_tokens=start_text, max_new_tokens=max_new_tokens,
                                            temperature=temp, top_k=top_k, top_p=top_p,
                                            repetition_penalty=repetition_penalty,
                                            device=device)
                    print(f"\n📝 Сгенерированный текст:\n{start_text}{generated}") # Выводим начальный текст + сгенерированный
                except Exception as e:
                    logger.error(f"Ошибка при генерации текста: {e}")
                    print(f"❌ Ошибка при генерации текста: {e}")
            elif command == "train":
                # Обновленный ввод данных для обучения
                data_source = input("Введите путь к текстовому файлу или URL веб-страницы: ").strip()
                if not data_source:
                    print("❌ Путь к файлу или URL не указан")
                    continue
                # Проверка, является ли это URL
                parsed_url = urlparse(data_source)
                is_url = parsed_url.scheme and parsed_url.netloc
                text = ""
                if is_url:
                    # Это URL
                    try:
                        print(f"🔄 Загрузка текста с URL: {data_source}")
                        text = load_text_from_url(data_source)
                        if not text:
                             print("❌ Не удалось загрузить или извлечь текст с указанного URL.")
                             continue
                    except Exception as e:
                        print(f"❌ Ошибка при загрузке с URL: {e}")
                        continue
                else:
                    # Предполагаем, что это путь к файлу
                    if not os.path.exists(data_source):
                        print(f"❌ Файл {data_source} не найден")
                        continue
                    try:
                        print(f"🔄 Загрузка текста из файла: {data_source}")
                        text = load_text(data_source)
                    except Exception as e:
                        print(f"❌ Ошибка при загрузке файла: {e}")
                        continue
                # 1. Определить профиль устройства
                hw_profile = detect_hardware_profile()
                print(f"Обнаружен профиль устройства: {hw_profile}")
                # 2. Получить адаптивные конфигурации
                # --- Используем обновленную конфигурацию для low_memory_cpu ---
                adaptive_config = ADAPTIVE_CONFIGS.get(hw_profile["profile_name"], ADAPTIVE_CONFIGS["low_memory_cpu"]) # fallback
                # ------------------------------------------------------------
                try:
                    token_type = input(f"Тип токенизации (bpe для tokenizers, по умолчанию {adaptive_config['token_type']}): ").strip().lower()
                    if token_type != "bpe":
                        print("⚠️  В этой версии поддерживается только 'bpe' с tokenizers.")
                        token_type = "bpe"
                    # Запрос параметров обучения с адаптивными значениями по умолчанию
                    try:
                        epochs = int(input(f"Количество эпох (по умолчанию {adaptive_config['epochs']}): ") or str(adaptive_config['epochs']))
                    except ValueError:
                        epochs = adaptive_config['epochs']
                    try:
                        seq_length = int(input(f"Длина последовательности (по умолчанию {adaptive_config['seq_length']}): ") or str(adaptive_config['seq_length']))
                    except ValueError:
                        seq_length = adaptive_config['seq_length']
                    try:
                        batch_size = int(input(f"Размер батча (по умолчанию {adaptive_config['batch_size']}): ") or str(adaptive_config['batch_size']))
                    except ValueError:
                        batch_size = adaptive_config['batch_size']
                    try:
                        learning_rate = float(input(f"Learning rate (по умолчанию {adaptive_config['learning_rate']}): ") or str(adaptive_config['learning_rate']))
                    except ValueError:
                        learning_rate = adaptive_config['learning_rate']
                    try:
                        vocab_size_input = int(input(f"Размер словаря для BPE (по умолчанию 30000): ") or "30000")
                        vocab_size_input = max(1000, vocab_size_input)
                    except ValueError:
                        vocab_size_input = 30000
                    
                    # --- Запрос параметров сжатия ---
                    try:
                        n_clusters_for_compression = int(input(f"Число кластеров для VQ сжатия (по умолчанию {DEFAULT_COMPRESSION_N_CLUSTERS}): ") or str(DEFAULT_COMPRESSION_N_CLUSTERS))
                        n_clusters_for_compression = max(2, min(65536, n_clusters_for_compression)) # Ограничение
                    except ValueError:
                        n_clusters_for_compression = DEFAULT_COMPRESSION_N_CLUSTERS
                    # -------------------------------
                    
                    print("🔄 Очистка текста...")
                    text = clean_text(text)
                    # --- Потоковая обработка ---
                    # Для потоковой обработки сохраняем текст во временный файл
                    temp_text_file_for_streaming = os.path.join(CACHE_DIR, f"temp_streaming_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    with open(temp_text_file_for_streaming, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print("🔄 Обучение BPE токенайзера...")
                    # --- Создание оптимального файла для обучения токенизатора ---
                    # Автоматически создаем временный файл с первыми 50 МБ оригинального текста
                    temp_tokenizer_train_file = os.path.join(CACHE_DIR, "temp_tokenizer_train_sample.txt")
                    max_sample_size_bytes = 50 * 1024 * 1024 # 50 MB
                    bytes_written = 0
                    with open(temp_text_file_for_streaming, 'r', encoding='utf-8', errors='ignore') as source_file, \
                         open(temp_tokenizer_train_file, 'w', encoding='utf-8') as sample_file:
                        while bytes_written < max_sample_size_bytes:
                            chunk = source_file.read(1024 * 1024) # Читаем по 1MB
                            if not chunk:
                                break
                            sample_file.write(chunk)
                            bytes_written += len(chunk.encode('utf-8'))
                    logger.info(f"Создан временный файл для обучения токенизатора: {temp_tokenizer_train_file} (размер: {bytes_written} байт)")
                    # Обучение токенайзера на уменьшенном образце
                    current_tokenizer = train_tokenizer([temp_tokenizer_train_file], vocab_size=vocab_size_input)
                    # --- Обновление постоянного токенайзера ---
                    current_tokenizer.save(PERSISTENT_TOKENIZER_PATH)
                    logger.info(f"✅ Постоянный токенайзер обновлён: {PERSISTENT_TOKENIZER_PATH}")
                    print(f"✅ Постоянный токенайзер обновлён. Размер словаря: {current_tokenizer.get_vocab_size()}")
                    # -------------------------------------
                    vocab_size = current_tokenizer.get_vocab_size()
                    # --- Улучшенное создание валидационного датасета ---
                    print("🔄 Подготовка данных с потоковой обработкой...")
                    # Создаем тренировочный датасет с потоковой обработкой
                    train_dataset = StreamingTextIterableDataset(
                        temp_text_file_for_streaming,
                        current_tokenizer,
                        seq_length,
                        stride=seq_length // 2
                    )
                    # Для валидации используем отдельный файл или подходящую часть.
                    # Здесь мы создадим отдельный файл для валидации, содержащий последние 10% оригинального текста.
                    # Это более надежный подход, чем использование первых токенов.
                    val_size_chars = max(1000, len(text) // 10) # Примерно 10% или минимум 1000 символов
                    val_text = text[-val_size_chars:]
                    temp_val_text_file = os.path.join(CACHE_DIR, "temp_val_text.txt")
                    with open(temp_val_text_file, 'w', encoding='utf-8') as f:
                        f.write(val_text)
                    val_dataset = StreamingTextIterableDataset(
                        temp_val_text_file,
                        current_tokenizer,
                        seq_length,
                        stride=seq_length // 2
                    )
                    logger.info(f"Создан отдельный файл валидации: {temp_val_text_file} (размер: {val_size_chars} символов)")
                    # Создаем DataLoader-ы
                    # Для тренировки используем IterableDataset
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        # num_workers можно установить > 0 благодаря улучшенному разделению
                        num_workers=2, # Пример: использовать 2 воркера
                        # collate_fn не нужен, так как данные уже тензоры
                    )
                    # Для валидации также используем IterableDataset
                    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        # num_workers также можно использовать
                        num_workers=1 # Один воркер для валидации
                    )
                    print("🔄 Создание модели...")
                    # Используем адаптивные параметры для модели
                    current_model = ModernGPT(
                        vocab_size=vocab_size,
                        hidden_size=adaptive_config['hidden_size'],
                        num_layers=adaptive_config['num_layers'],
                        num_heads=adaptive_config['num_heads'],
                        ff_hidden_size=adaptive_config['ff_hidden_size'],
                        dropout=adaptive_config['dropout'],
                        max_seq_length=seq_length # Убедиться, что это передается
                    )
                    print(current_model.get_model_info())
                    criterion = LabelSmoothingLoss(smoothing=0.1)
                    optimizer = optim.AdamW(current_model.parameters(), lr=learning_rate, weight_decay=0.01)
                    current_token_type = token_type
                    current_model_type = "gpt"
                    print("🔄 Начало обучения...")
                    model_path, metrics_collector = train_model(current_model, train_loader, val_loader, criterion, optimizer,
                                                              epochs, device,
                                                              PERSISTENT_TOKENIZER_PATH, vocab_size, # Используем постоянный путь
                                                              token_type, learning_rate, current_model_type,
                                                              n_clusters_for_compression=n_clusters_for_compression) # Передаем параметр сжатия
                    if model_path:
                        print(f"✅ Обучение завершено! Модель (сжатая) сохранена в {os.path.basename(model_path)}")
                        print(f"📊 Метрики сохранены в {METRICS_DIR}")
                        # Загружаем модель после обучения, чтобы она была готова к использованию
                        # Также пытаемся загрузить сжатую версию, если она была создана
                        loaded_model, loaded_tokenizer_path, loaded_token_type, loaded_model_type, loaded_perplexity, loaded_weight_stats, loaded_training_config = load_model_with_dicts(model_path, device)
                        if loaded_model is not None:
                            current_model = loaded_model
                            current_token_type = loaded_token_type
                            current_model_type = loaded_model_type
                            current_perplexity = loaded_perplexity
                            current_weight_stats = loaded_weight_stats
                            current_training_config = loaded_training_config
                            # Перезагружаем токенайзер из постоянного файла
                            if os.path.exists(PERSISTENT_TOKENIZER_PATH):
                                current_tokenizer = Tokenizer.from_file(PERSISTENT_TOKENIZER_PATH)
                            print("✅ Модель и постоянный токенайзер перезагружены после обучения.")
                    else:
                        print("⚠️  Обучение завершено, но модель не была сохранена")
                    # Очистка временных файлов
                    temp_files_to_cleanup = [
                        temp_text_file_for_streaming, 
                        temp_tokenizer_train_file, # Удаляем файл для обучения токенизатора
                        temp_val_text_file
                    ]
                    for temp_file in temp_files_to_cleanup:
                         if os.path.exists(temp_file):
                             try:
                                 os.remove(temp_file)
                                 logger.debug(f"Удален временный файл: {temp_file}")
                             except Exception as rm_e:
                                 logger.warning(f"Не удалось удалить временный файл {temp_file}: {rm_e}")
                except Exception as e:
                    logger.error(f"Ошибка при обучении: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    print(f"❌ Ошибка при обучении: {e}")
                    # Очистка временных файлов в случае ошибки
                    temp_files_to_cleanup = [
                        temp_text_file_for_streaming if 'temp_text_file_for_streaming' in locals() else None,
                        temp_tokenizer_train_file if 'temp_tokenizer_train_file' in locals() else None, # Удаляем файл для обучения токенизатора
                        temp_val_text_file if 'temp_val_text_file' in locals() else None
                    ]
                    for temp_file in temp_files_to_cleanup:
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception as rm_e:
                                logger.warning(f"Не удалось удалить временный файл {temp_file}: {rm_e}")
            elif command == "save":
                if current_model is None:
                    print("❌ Нет модели для сохранения")
                    continue
                # --- Сохраняем только сжатую модель ---
                model_path = save_compressed_model_with_timestamp(current_model,
                                                     PERSISTENT_TOKENIZER_PATH, # Используем постоянный путь
                                                     vocab_size, token_type=current_token_type,
                                                     model_type=current_model_type,
                                                     perplexity=current_perplexity,
                                                     training_config=current_training_config)
                # --- Сохраняем конфигурацию отдельно ---
                if model_path:
                    config_data_to_save = {
                        'model_config': {
                            'hidden_size': getattr(current_model, 'hidden_size', 512),
                            'num_layers': getattr(current_model, 'num_layers', 6),
                            'num_heads': getattr(current_model, 'blocks', [None])[0].attention.num_heads if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 8,
                            'ff_hidden_size': getattr(current_model, 'blocks', [None])[0].ffn.linear1.out_features if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 2048,
                            'dropout': getattr(current_model, 'blocks', [None])[0].dropout1.p if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 0.1,
                            'model_type': type(current_model).__name__,
                            'vocab_size': vocab_size
                        },
                        'training_config': current_training_config
                    }
                    config_path = model_path.replace(".pth", ".json")
                    try:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data_to_save, f, indent=2, ensure_ascii=False)
                        logger.info(f"Конфигурация модели сохранена в {config_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении конфигурации модели: {e}")
                # --------------------------------------
                if model_path:
                    print(f"✅ Модель (сжатая) сохранена в {os.path.basename(model_path)}")
                else:
                    print("❌ Ошибка при сохранении модели")
            elif command == "load":
                model_files = list_available_models()
                if not model_files:
                    continue
                try:
                    choice = int(input("Выберите модель (номер): ")) - 1
                    if 0 <= choice < len(model_files):
                        model_path = model_files[choice]
                        # Загружаем сжатую модель
                        loaded_model, loaded_tokenizer_path, loaded_token_type, loaded_model_type, loaded_perplexity, loaded_weight_stats, loaded_training_config = load_model_with_dicts(model_path, device)
                        if loaded_model is not None:
                            current_model = loaded_model
                            current_token_type = loaded_token_type
                            current_model_type = loaded_model_type
                            current_perplexity = loaded_perplexity
                            current_weight_stats = loaded_weight_stats
                            current_training_config = loaded_training_config
                            vocab_size = loaded_training_config.get('vocab_size', 0) # Получаем из чекпойнта
                            print(f"✅ Модель загружена из {os.path.basename(model_path)}")
                            print(f"   Тип токенизации: {current_token_type}")
                            print(f"   Тип модели: {current_model_type}")
                            if current_perplexity:
                                print(f"   Perplexity: {current_perplexity:.4f}")
                            # Загрузка токенайзера из постоянного файла
                            if os.path.exists(PERSISTENT_TOKENIZER_PATH):
                                current_tokenizer = Tokenizer.from_file(PERSISTENT_TOKENIZER_PATH)
                                print(f"✅ Постоянный токенайзер загружен из {PERSISTENT_TOKENIZER_PATH}")
                            else:
                                print("⚠️  Постоянный токенайзер не найден.")
                                current_tokenizer = None
                        else:
                            print("❌ Ошибка при загрузке модели")
                    else:
                        print("❌ Неверный номер модели")
                except ValueError:
                    print("❌ Неверный ввод")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке модели: {e}")
                    print(f"❌ Ошибка при загрузке модели: {e}")
            elif command == "list":
                list_available_models()
            elif command == "info":
                if current_model is None:
                    print("❌ Нет загруженной модели")
                    continue
                print("\nИнформация о текущей модели:")
                print(current_model.get_model_info())
                print(f"  Vocabulary size (loaded): {vocab_size}")
                print(f"  Token type: {current_token_type}")
                print(f"  Model type: {current_model_type}")
                print(f"  Device: {device}")
                if current_perplexity:
                    print(f"  Perplexity: {current_perplexity:.4f}")
                print(f"  Models directory: {os.path.abspath(MODELS_DIR)}")
                print(f"  Metrics directory: {os.path.abspath(METRICS_DIR)}")
                print(f"  Cache directory: {os.path.abspath(CACHE_DIR)}")
                print(f"  Tokenizer path: {PERSISTENT_TOKENIZER_PATH}") # Показываем постоянный путь
                if current_training_config:
                    print("\nКонфигурация обучения:")
                    for key, value in current_training_config.items():
                        print(f"  {key}: {value}")
            elif command == "weights":
                if current_model is None:
                    print("❌ Нет загруженной модели")
                    continue
                print("\n📊 Статистики весов модели:")
                if current_weight_stats:
                    for layer_name, stats in current_weight_stats.items():
                        print(f"\n{layer_name}:")
                        print(f"  Shape: {stats['shape']}")
                        print(f"  Mean: {stats['mean']:.6f}")
                        print(f"  Std: {stats['std']:.6f}")
                        print(f"  Min: {stats['min']:.6f}")
                        print(f"  Max: {stats['max']:.6f}")
                else:
                    print("Нет доступных статистик весов")
            elif command == "metrics":
                print(f"\n📊 Директория с метриками: {os.path.abspath(METRICS_DIR)}")
                metrics_files = glob.glob(os.path.join(METRICS_DIR, "*.json"))
                if metrics_files:
                    print("Доступные файлы метрик:")
                    for i, metric_file in enumerate(sorted(metrics_files, key=os.path.getctime, reverse=True)[:10]):
                        timestamp = datetime.fromtimestamp(os.path.getctime(metric_file)).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"  {i+1}. {os.path.basename(metric_file)} ({timestamp})")
                else:
                    print("Нет доступных файлов метрик")
            elif command == "chat":
                 chat_with_sin(current_model, current_tokenizer, device)
            else:
                print("❓ Неизвестная команда. Доступные команды:")
                print("  generate, train, save, load, list, info, weights, metrics, chat, quit")
        except KeyboardInterrupt:
            print("\n⚠️  Прерывание программы...")
            if current_model is not None:
                print("Автоматическое сохранение модели...")
                # --- Сохраняем только сжатую модель ---
                model_path = save_compressed_model_with_timestamp(current_model,
                                        PERSISTENT_TOKENIZER_PATH, # Используем постоянный путь
                                        vocab_size, token_type=current_token_type,
                                        model_type=current_model_type,
                                        perplexity=current_perplexity,
                                        training_config=current_training_config)
                # --- Сохраняем конфигурацию отдельно ---
                if model_path:
                    config_data_to_save = {
                        'model_config': {
                            'hidden_size': getattr(current_model, 'hidden_size', 512),
                            'num_layers': getattr(current_model, 'num_layers', 6),
                            'num_heads': getattr(current_model, 'blocks', [None])[0].attention.num_heads if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 8,
                            'ff_hidden_size': getattr(current_model, 'blocks', [None])[0].ffn.linear1.out_features if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 2048,
                            'dropout': getattr(current_model, 'blocks', [None])[0].dropout1.p if hasattr(current_model, 'blocks') and len(current_model.blocks) > 0 else 0.1,
                            'model_type': type(current_model).__name__,
                            'vocab_size': vocab_size
                        },
                        'training_config': current_training_config
                    }
                    config_path = model_path.replace(".pth", ".json")
                    try:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data_to_save, f, indent=2, ensure_ascii=False)
                        logger.info(f"Конфигурация модели сохранена в {config_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении конфигурации модели: {e}")
                # --------------------------------------
                if model_path:
                    print(f"✅ Модель (сжатая) сохранена в {os.path.basename(model_path)}")
                else:
                    print("❌ Ошибка при сохранении модели")
            print("До свидания!")
            break
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    print(f"🤖 Современный генеративный ИИ '{ASSISTANT_NAME}' с GPT-архитектурой (оптимизированная версия)")
    print("Поддерживаемые форматы файлов: .txt, .docx, .pdf, .json")
    print("Оптимизации для CPU: уменьшенные параметры модели, улучшенная обработка ошибок")
    print("Интеграция с tokenizers для BPE.")
    print("Поддержка обучения на тексте из веб-страниц (URL).")
    print("Потоковая обработка больших файлов для экономии памяти.")
    print("Поддержка обучения на JSON-датасетах (например, SiberiaSoft/SiberianPersonaChat).")
    print("Интеграция VQ-сжатия моделей. Теперь сохраняются и загружаются только сжатые модели.")
    print(f"📁 Модели сохраняются в: {os.path.abspath(MODELS_DIR)}")
    print(f"📝 Логи сохраняются в: {os.path.abspath(LOGS_DIR)}")
    print(f"📊 Метрики сохраняются в: {os.path.abspath(METRICS_DIR)}")
    print(f"Кэширование в: {os.path.abspath(CACHE_DIR)}")
    print(f"Постоянный токенайзер: {os.path.abspath(PERSISTENT_TOKENIZER_PATH)}")
    print("\n🚀 Запуск интерактивного режима...")
    interactive_mode()
