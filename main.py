import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
import os
import re
import glob
import time
import math
from datetime import datetime
from collections import Counter, defaultdict
import logging
from torch.nn.utils.rnn import pad_sequence
import json
import matplotlib.pyplot as plt
import psutil  # Для мониторинга системных ресурсов
import gc     # Для ручной очистки памяти
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
# Создание директории для моделей
MODELS_DIR = "models"
LOGS_DIR = "logs"
METRICS_DIR = "metrics"
CACHE_DIR = "cache"
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
DEFAULT_HIDDEN_SIZE = 512  # Уменьшено с 768
DEFAULT_NUM_LAYERS = 6     # Уменьшено с 12
DEFAULT_ATTENTION_HEADS = 8 # Уменьшено с 12
DEFAULT_FF_HIDDEN_SIZE = 2048 # Уменьшено с 3072
DEFAULT_DROPOUT = 0.1
MAX_SAVED_MODELS = 5
DEFAULT_TOKEN_TYPE = "bpe"
DEFAULT_MODEL_TYPE = "gpt"

# ------------------
# Адаптивные конфигурации
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
    },
    "low_memory_cpu": {
        "seq_length": 64,
        "batch_size": 1,
        "epochs": 10,
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "ff_hidden_size": 512,
        "dropout": 0.1,
        "learning_rate": 3e-4,
    }
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
    def generate(self, input_ids, max_new_tokens, temperature=1.0, do_sample=True):
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
                if next_token.item() == 0:  # Assuming 0 is end token
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
# Подготовка данных
# ------------------
def load_text(file_path):
    logger.info(f"Попытка загрузки файла: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.txt':
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
def tokenize_text(text, token_type="bpe"):
    if token_type == "char":
        return list(text)
    elif token_type == "word":
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return words
    elif token_type == "bpe":
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return words
    else:
        raise ValueError(f"Неизвестный тип токенизации: {token_type}")
def clean_text(text):
    original_length = len(text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\n\u0400-\u04FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    cleaned_length = len(text)
    logger.info(f"Текст очищен: {original_length} -> {cleaned_length} символов")
    return text.strip()
def create_token_mappings(tokens):
    unique_tokens = sorted(list(set(tokens)))
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    all_tokens = special_tokens + unique_tokens
    token_to_idx = {token: i for i, token in enumerate(all_tokens)}
    idx_to_token = {i: token for i, token in enumerate(all_tokens)}
    logger.info(f"Созданы словари: {len(all_tokens)} токенов (включая специальные)")
    return token_to_idx, idx_to_token, len(all_tokens)
def create_sequences(tokens, token_to_idx, seq_length, stride=None):
    if stride is None:
        stride = seq_length // 2
    if len(tokens) < seq_length + 1:
        raise ValueError(f"Текст слишком короткий. Минимальная длина: {seq_length + 1}, текущая: {len(tokens)}")
    bos_token = token_to_idx['<BOS>']
    eos_token = token_to_idx['<EOS>']
    data = [bos_token] + [token_to_idx.get(token, token_to_idx['<UNK>']) for token in tokens] + [eos_token]
    X, y = [], []
    for i in range(0, len(data) - seq_length, stride):
        if i + seq_length + 1 <= len(data):
            X.append(data[i:i+seq_length])
            y.append(data[i+1:i+seq_length+1])
    logger.info(f"Создано {len(X)} последовательностей длиной {seq_length} с шагом {stride}")
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
# ------------------
# Управление моделями
# ------------------
def get_model_files():
    model_files = glob.glob(os.path.join(MODELS_DIR, "gpt_model_*.pth"))
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
def save_model_with_timestamp(model, token_to_idx, idx_to_token, vocab_size, 
                            loss=0.0, token_type="bpe", perplexity=None,
                            training_config=None, metrics_collector=None, model_type="gpt"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"gpt_model_{timestamp}.pth"
    model_path = os.path.join(MODELS_DIR, model_filename)
    try:
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
        torch.save({
            'model_state_dict': model.state_dict(),
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'vocab_size': vocab_size,
            'timestamp': timestamp,
            'loss': loss,
            'perplexity': perplexity,
            'token_type': token_type,
            'model_type': model_type,
            'weight_statistics': weight_stats,
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
        logger.info(f"Модель сохранена: {model_path}")
        if metrics_collector:
            metrics_filename = f"metrics_{timestamp}.json"
            metrics_collector.save_metrics(metrics_filename)
            metrics_collector.plot_metrics(f"metrics_{timestamp}")
        cleanup_old_models()
        return model_path
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")
        return None
def load_model_with_dicts(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {})
        model_type = checkpoint.get('model_type', 'gpt')
        if model_type == 'gpt':
            model = ModernGPT(
                checkpoint['vocab_size'],
                model_config.get('hidden_size', 512),
                model_config.get('num_layers', 6),
                model_config.get('num_heads', 8),
                model_config.get('ff_hidden_size', 2048),
                dropout=model_config.get('dropout', 0.1)
            )
        else:
            model = ModernGPT(
                checkpoint['vocab_size'],
                model_config.get('hidden_size', 512),
                model_config.get('num_layers', 6),
                model_config.get('num_heads', 8),
                model_config.get('ff_hidden_size', 2048),
                dropout=model_config.get('dropout', 0.1)
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        token_to_idx = checkpoint['token_to_idx']
        idx_to_token = checkpoint['idx_to_token']
        timestamp = checkpoint.get('timestamp', 'unknown')
        loss = checkpoint.get('loss', 0.0)
        token_type = checkpoint.get('token_type', 'bpe')
        model_type = checkpoint.get('model_type', 'gpt')
        perplexity = checkpoint.get('perplexity', None)
        weight_stats = checkpoint.get('weight_statistics', {})
        training_config = checkpoint.get('training_config', {})
        logger.info(f"Модель загружена: {model_path} (timestamp: {timestamp}, loss: {loss:.4f})")
        return model, token_to_idx, idx_to_token, token_type, model_type, perplexity, weight_stats, training_config
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_path}: {e}")
        return None, None, None, None, None, None, None, None
def list_available_models():
    model_files = get_model_files()
    if not model_files:
        print("Нет доступных моделей")
        return []
    print("\nДоступные модели:")
    for i, model_file in enumerate(model_files):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            timestamp = checkpoint.get('timestamp', 'unknown')
            loss = checkpoint.get('loss', 0.0)
            vocab_size = checkpoint.get('vocab_size', 0)
            token_type = checkpoint.get('token_type', 'bpe')
            model_type = checkpoint.get('model_type', 'gpt')
            perplexity = checkpoint.get('perplexity', 'N/A')
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
# Генерация текста
# ------------------
def generate_text(model, token_to_idx, idx_to_token, start_tokens, 
                 max_new_tokens=200, temperature=1.0, top_k=0, top_p=1.0, 
                 repetition_penalty=1.0, device='cpu', token_type="bpe"):
    logger.info(f"Начало генерации текста: '{start_tokens}', max_new_tokens: {max_new_tokens}")
    logger.info(f"Параметры: температура={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}")
    model.eval()
    with torch.no_grad():
        tokens = tokenize_text(start_tokens, token_type) if token_type != "char" else list(start_tokens)
        input_ids = [token_to_idx.get('<BOS>', 0)]
        input_ids.extend([token_to_idx.get(token, token_to_idx.get('<UNK>', 1)) for token in tokens])
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated_tokens = []
        for _ in range(max_new_tokens):
            output, _ = model(input_ids)
            next_token_logits = output[0, -1, :]
            probs = advanced_sampling(
                next_token_logits.unsqueeze(0), 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                previous_tokens=input_ids[0].tolist()
            )
            try:
                next_token = torch.multinomial(probs, 1)[0, 0].item()
                if next_token == token_to_idx.get('<EOS>', -1):
                    break
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            except Exception as e:
                logger.error(f"Ошибка при генерации токена: {e}")
                break
        generated_text_tokens = [idx_to_token.get(idx, '?') for idx in generated_tokens]
        if token_type == "word" or token_type == "bpe":
            generated_text = ' '.join(generated_text_tokens)
        else:
            generated_text = ''.join(generated_text_tokens)
        logger.info(f"Генерация завершена, сгенерировано {len(generated_tokens)} токенов")
        return start_tokens + generated_text
# ------------------
# Обучение с улучшенной обработкой ошибок
# ------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
                token_to_idx, idx_to_token, vocab_size, token_type="bpe", 
                learning_rate=DEFAULT_LEARNING_RATE, model_type="gpt"):
    logger.info(f"Начало обучения модели на устройстве {device}")
    logger.info(f"Параметры обучения: epochs={epochs}, batch_size={train_loader.batch_size}")
    metrics_collector = MetricsCollector()
    training_config = {
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': learning_rate,
        'seq_length': train_loader.dataset[0][0].size(0) if len(train_loader.dataset) > 0 else 0,
        'vocab_size': vocab_size,
        'token_type': token_type,
        'model_type': model_type,
        'device': str(device),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    model.to(device)
    model.train()
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
            total_batches = len(train_loader)
            logger.info(f"Эпоха {epoch+1}/{epochs} начата")
            # Training phase
            model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                try:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
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
                    metrics_collector.add_batch_loss(loss.item())
                    # Логирование каждые 10% батчей
                    if batch_idx % max(1, total_batches // 10) == 0 and batch_idx > 0:
                        avg_batch_loss = total_loss / (batch_idx + 1)
                        logger.info(f"Эпоха {epoch+1}/{epochs}, Батч {batch_idx}/{total_batches}, Loss: {avg_batch_loss:.4f}")
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
                    continue
            # Validation phase
            model.eval()
            val_loss = 0
            val_samples = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    try:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        output, _ = model(x_batch)
                        loss = criterion(output.reshape(-1, output.size(-1)), y_batch.reshape(-1))
                        val_loss += loss.item() * x_batch.size(0) * x_batch.size(1)
                        val_samples += x_batch.size(0) * x_batch.size(1)
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
            # Сохранение лучшей модели
            if val_loss < best_loss:
                best_loss = val_loss
                best_perplexity = val_perplexity
                metrics_collector.collect_weight_statistics(model)
                model_path = save_model_with_timestamp(model, token_to_idx, idx_to_token, 
                                                     vocab_size, val_loss, token_type, val_perplexity,
                                                     training_config, metrics_collector, model_type)
                if model_path:
                    best_model_path = model_path
                    logger.info(f"Новая лучшая модель сохранена: loss {val_loss:.4f}, perplexity {val_perplexity:.4f}")
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
    token_to_idx = None
    idx_to_token = None
    vocab_size = 0
    current_token_type = "bpe"
    current_model_type = "gpt"
    current_perplexity = None
    current_weight_stats = {}
    current_training_config = {}
    print("\n" + "="*80)
    print("🤖 Современный генеративный ИИ с GPT-архитектурой (оптимизированная версия)")
    print("="*80)
    print(f"📁 Модели сохраняются в: {os.path.abspath(MODELS_DIR)}")
    print(f"📝 Логи сохраняются в: {os.path.abspath(LOGS_DIR)}")
    print(f"📊 Метрики сохраняются в: {os.path.abspath(METRICS_DIR)}")
    print(f"Кэширование в: {os.path.abspath(CACHE_DIR)}")
    print("\nДоступные команды:")
    print("  generate     - Генерация текста (продвинутая)")
    print("  train        - Обучение модели")
    print("  save         - Сохранение текущей модели")
    print("  load         - Загрузка модели")
    print("  list         - Список доступных моделей")
    print("  info         - Информация о текущей модели")
    print("  weights      - Просмотр статистик весов")
    print("  metrics      - Просмотр метрик обучения")
    print("  quit         - Выход")
    print("="*80)
    while True:
        try:
            command = input("\nВведите команду: ").strip().lower()
            if command == "quit":
                if current_model is not None:
                    print("Автоматическое сохранение модели...")
                    save_model_with_timestamp(current_model, token_to_idx, idx_to_token, 
                                            vocab_size, token_type=current_token_type,
                                            model_type=current_model_type,
                                            perplexity=current_perplexity,
                                            training_config=current_training_config)
                print("До свидания!")
                break
            elif command == "generate":
                if current_model is None:
                    print("❌ Нет загруженной модели. Сначала загрузите или обучите модель.")
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
                    generated = generate_text(current_model, token_to_idx, idx_to_token, 
                                            start_tokens=start_text, max_new_tokens=max_new_tokens,
                                            temperature=temp, top_k=top_k, top_p=top_p,
                                            repetition_penalty=repetition_penalty,
                                            device=device, token_type=current_token_type)
                    print(f"\n📝 Сгенерированный текст:\n{generated}")
                except Exception as e:
                    logger.error(f"Ошибка при генерации текста: {e}")
                    print(f"❌ Ошибка при генерации текста: {e}")
            elif command == "train":
                file_path = input("Введите путь к текстовому файлу: ").strip()
                if not file_path:
                    print("❌ Путь к файлу не указан")
                    continue
                if not os.path.exists(file_path):
                    print(f"❌ Файл {file_path} не найден")
                    continue
                
                # 1. Определить профиль устройства
                hw_profile = detect_hardware_profile()
                print(f"Обнаружен профиль устройства: {hw_profile}")
                
                # 2. Получить адаптивные конфигурации
                adaptive_config = ADAPTIVE_CONFIGS.get(hw_profile["profile_name"], ADAPTIVE_CONFIGS["low_memory_cpu"]) # fallback
                
                try:
                    token_type = input(f"Тип токенизации (char/word/bpe, по умолчанию {adaptive_config['token_type'] if 'token_type' in adaptive_config else 'bpe'}): ").strip().lower()
                    if token_type not in ["char", "word", "bpe"]:
                        token_type = adaptive_config.get('token_type', 'bpe')
                    
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
                    
                    print("🔄 Загрузка текста...")
                    text = load_text(file_path)
                    text = clean_text(text)
                    tokens = tokenize_text(text, token_type)
                    print(f"Текст токенизирован: {len(tokens)} токенов")
                    if len(tokens) < seq_length:
                        print("❌ Текст слишком короткий для обучения")
                        continue
                    token_to_idx, idx_to_token, vocab_size = create_token_mappings(tokens)
                    X, y = create_sequences(tokens, token_to_idx, seq_length)
                    if len(X) == 0:
                        print("❌ Недостаточно данных для обучения")
                        continue
                    dataset = torch.utils.data.TensorDataset(X, y)
                    train_size = int(0.9 * len(dataset))
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
                                                              epochs, device, token_to_idx, idx_to_token, vocab_size, 
                                                              token_type, learning_rate, current_model_type)
                    if model_path:
                        print(f"✅ Обучение завершено! Модель сохранена в {os.path.basename(model_path)}")
                        print(f"📊 Метрики сохранены в {METRICS_DIR}")
                    else:
                        print("⚠️  Обучение завершено, но модель не была сохранена")
                except Exception as e:
                    logger.error(f"Ошибка при обучении: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    print(f"❌ Ошибка при обучении: {e}")
            elif command == "save":
                if current_model is None:
                    print("❌ Нет модели для сохранения")
                    continue
                model_path = save_model_with_timestamp(current_model, token_to_idx, idx_to_token, 
                                                     vocab_size, token_type=current_token_type,
                                                     model_type=current_model_type,
                                                     perplexity=current_perplexity,
                                                     training_config=current_training_config)
                if model_path:
                    print(f"✅ Модель сохранена в {os.path.basename(model_path)}")
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
                        loaded_model, loaded_token_to_idx, loaded_idx_to_token, loaded_token_type, loaded_model_type, loaded_perplexity, loaded_weight_stats, loaded_training_config = load_model_with_dicts(model_path, device)
                        if loaded_model is not None:
                            current_model = loaded_model
                            token_to_idx = loaded_token_to_idx
                            idx_to_token = loaded_idx_to_token
                            current_token_type = loaded_token_type
                            current_model_type = loaded_model_type
                            current_perplexity = loaded_perplexity
                            current_weight_stats = loaded_weight_stats
                            current_training_config = loaded_training_config
                            checkpoint = torch.load(model_path, map_location=device)
                            vocab_size = checkpoint.get('vocab_size', len(token_to_idx))
                            print(f"✅ Модель загружена из {os.path.basename(model_path)}")
                            print(f"   Тип токенизации: {current_token_type}")
                            print(f"   Тип модели: {current_model_type}")
                            if current_perplexity:
                                print(f"   Perplexity: {current_perplexity:.4f}")
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
                if token_to_idx and idx_to_token:
                    print(f"  Dictionary size: {len(token_to_idx)} tokens")
                    sample_tokens = list(token_to_idx.keys())[:30]
                    print(f"  Sample tokens: {sample_tokens}")
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
            else:
                print("❓ Неизвестная команда. Доступные команды:")
                print("  generate, train, save, load, list, info, weights, metrics, quit")
        except KeyboardInterrupt:
            print("\n⚠️  Прерывание программы...")
            if current_model is not None:
                print("Автоматическое сохранение модели...")
                save_model_with_timestamp(current_model, token_to_idx, idx_to_token, 
                                        vocab_size, token_type=current_token_type,
                                        model_type=current_model_type,
                                        perplexity=current_perplexity,
                                        training_config=current_training_config)
            print("До свидания!")
            break
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"❌ Неожиданная ошибка: {e}")
if __name__ == "__main__":
    print("🤖 Современный генеративный ИИ с GPT-архитектурой (оптимизированная версия)")
    print("Поддерживаемые форматы файлов: .txt, .docx, .pdf")
    print("Оптимизации для CPU: уменьшенные параметры модели, улучшенная обработка ошибок")
    print(f"📁 Модели сохраняются в: {os.path.abspath(MODELS_DIR)}")
    print(f"📝 Логи сохраняются в: {os.path.abspath(LOGS_DIR)}")
    print(f"📊 Метрики сохраняются в: {os.path.abspath(METRICS_DIR)}")
    print(f"Кэширование в: {os.path.abspath(CACHE_DIR)}")
    print("\n🚀 Запуск интерактивного режима...")
    interactive_mode()
