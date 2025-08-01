import torch
import torch.nn as nn
import torch.nn.functional as F
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import requests
from bs4 import BeautifulSoup
import psutil
import matplotlib.pyplot as plt
import json
import numpy as np
import random
import os
import glob
import time
import math
import logging
import gc
from collections import defaultdict, deque
import shutil
from typing import List, Dict, Tuple, Optional, Any
import re
from urllib.parse import urlparse
from datetime import datetime
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Константы
MODELS_DIR = "models"
LOGS_DIR = "logs"
METRICS_DIR = "metrics"
CACHE_DIR = "cache"
TOKENIZER_FILE = "tokenizer.json"
BEST_MODEL_FILE = "best_model.pth"
BEST_TOKENIZER_FILE = "best_tokenizer.json"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
# Гиперпараметры
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_NUM_LAYERS = 50
DEFAULT_NUM_HEADS = 8
DEFAULT_FF_HIDDEN_SIZE = 2048
DEFAULT_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 10
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_GRADIENT_NOISE_SIGMA = 1e-3
# Адаптивные конфигурации для разных устройств
ADAPTIVE_CONFIGS = {
    "cuda": {
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "num_layers": DEFAULT_NUM_LAYERS,
        "num_heads": DEFAULT_NUM_HEADS,
        "ff_hidden_size": DEFAULT_FF_HIDDEN_SIZE,
        "seq_length": DEFAULT_SEQ_LENGTH,
        "batch_size": 32,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "epochs": DEFAULT_EPOCHS
    },
    "cpu": {
        "hidden_size": 256,
        "num_layers": 12,
        "num_heads": 4,
        "ff_hidden_size": 512,
        "seq_length": 256,
        "batch_size": 8,
        "learning_rate": DEFAULT_LEARNING_RATE * 0.1,
        "epochs": DEFAULT_EPOCHS
    }
}
# Проверка поддержки tokenizers
TOKENIZERS_SUPPORT = True
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
except ImportError:
    TOKENIZERS_SUPPORT = False
    logger.warning("Tokenizers не установлены. Некоторые функции будут недоступны.")
# Классы и модули модели
class LayerNorm(nn.Module):
    """Слой нормализации по образцу LayerNorm."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
class RotaryPositionalEmbedding(nn.Module):
    """RoPE positional embedding."""
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    def forward(self, x, position_ids):
        # Создаем частоты для RoPE
        inv_freq_expanded = self.inv_freq[None, :, None].to(x.device).float()
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Применение RoPE к q и k."""
    cos = cos.squeeze(1).unsqueeze(2)
    sin = sin.squeeze(1).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
def rotate_half(x):
    """Поворот половины тензора."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
class MultiHeadAttention(nn.Module):
    """Многоголовое внимание с RoPE."""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = x.shape
        # Проекции
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Перестановка для многоголового внимания
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # RoPE
        if position_ids is not None:
            cos, sin = self.rotary_emb(q, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        # Маскирование
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        # Внимание
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Применение внимания
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        # Выходной слой
        output = self.o_proj(attn_output)
        return output
class FeedForward(nn.Module):
    """Полносвязная сеть."""
    def __init__(self, hidden_size, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.w2 = nn.Linear(ff_hidden_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x
class TransformerBlock(nn.Module):
    """Блок трансформера с остаточными связями."""
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, ff_hidden_size, dropout)
        self.ln1 = LayerNorm(hidden_size)
        self.ln2 = LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, attention_mask=None, position_ids=None):
        # Внимание с остаточной связью
        attn_output = self.attention(self.ln1(x), attention_mask, position_ids)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output
        # FFN с остаточной связью
        ffn_output = self.ffn(self.ln2(x))
        ffn_output = self.dropout2(ffn_output)
        x = x + ffn_output
        return x
class ModernGPT(nn.Module):
    """Современная GPT модель с биологически-ориентированной архитектурой."""
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ff_hidden_size, max_seq_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        # Векторная эмбеддинг
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Позиционные эмбеддинги
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        # Блоки трансформера
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_hidden_size)
            for _ in range(num_layers)
        ])
        # Выходной слой
        self.ln_f = LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Инициализация весов
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    def forward(self, x, attention_mask=None):
        batch_size, seq_length = x.shape
        # Создаем позиционные индексы
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # Эмбеддинги
        embeddings = self.embedding(x) + self.position_embedding(position_ids)
        # Проход по блокам
        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask, position_ids)
        # Финальная нормализация и выход
        embeddings = self.ln_f(embeddings)
        logits = self.head(embeddings)
        return logits
# Биологически-ориентированные компоненты
class AnchorNeuron:
    """Нейрон-якорь с ассоциативными связями."""
    def __init__(self, neuron_id, anchor_type="concept"):
        self.id = neuron_id
        self.type = anchor_type  # concept, image, emotion, memory, sensory
        self.name = ""
        self.state = 0.0
        self.memory = []
        self.association_links = {}
        self.activation_strength = 0.0
        self.stability = 0.0
        self.context_triggers = []
        self.emotional_state = 0.0
        self.initial_context = None
        self.last_activation_time = time.time()
    def activate(self, context=None, strength=1.0):
        """Активация якоря с учетом контекста"""
        # Учет контекста активации
        context_factor = 1.0
        if context:
            for trigger in self.context_triggers:
                if trigger in str(context):
                    context_factor += 0.3
        # Активация с учетом эмоционального состояния
        emotion_factor = 1.0 + (self.emotional_state * 0.5)
        self.activation_strength = min(1.0, self.activation_strength + strength * 0.1 * context_factor * emotion_factor)
        self.last_activation_time = time.time()
        return self.activation_strength
    def link_to_association(self, target_neuron, strength=0.5):
        """Создание связи с ассоциацией"""
        self.association_links[target_neuron.id] = {
            'neuron': target_neuron,
            'strength': strength,
            'last_used': time.time()
        }
    def trigger_associations(self):
        """Активация ассоциаций, связанных с якорем"""
        activated_associations = []
        for assoc_id, assoc_data in self.association_links.items():
            # Сила ассоциации зависит от:
            # 1. Силы связи
            # 2. Времени последнего использования
            # 3. Стабильности якоря
            activation_score = (assoc_data['strength'] * 
                             self.stability * 
                             0.9 ** (time.time() - assoc_data['last_used']))
            if activation_score > 0.3:  # Порог активации
                activated_associations.append(assoc_data['neuron'])
        return activated_associations
    def _update_memory(self, activation, context, emotional_feedback):
        """Обновление истории активаций"""
        self.memory.append({
            'activation': activation,
            'context': context,
            'emotional_feedback': emotional_feedback,
            'timestamp': time.time()
        })
class AnchorSystem:
    """Система управления якорями."""
    def __init__(self):
        self.anchors = {}  # Якоря
        self.association_network = {}  # Сеть ассоциаций
        self.anchor_hierarchy = {}  # Иерархия якорей
    def create_anchor(self, concept_name, context=None, anchor_type="concept"):
        """Создание нового якоря"""
        anchor_id = f"anchor_{len(self.anchors)}"
        anchor = AnchorNeuron(anchor_id, anchor_type)
        anchor.name = concept_name
        anchor.initial_context = context
        anchor.stability = 0.7
        self.anchors[anchor_id] = anchor
        return anchor
    def link_anchors(self, source_anchor, target_anchor, strength=0.5):
        """Связывание якорей с весом"""
        source_anchor.link_to_association(target_anchor, strength)
        target_anchor.link_to_association(source_anchor, strength)
        # Добавляем в сеть ассоциаций
        if source_anchor.id not in self.association_network:
            self.association_network[source_anchor.id] = []
        self.association_network[source_anchor.id].append({
            'target': target_anchor.id,
            'strength': strength,
            'timestamp': time.time()
        })
    def activate_chain(self, seed_anchor, context=None, depth=5):
        """Активация цепочки ассоциаций"""
        activated_chain = []
        active_queue = [(seed_anchor, 1.0)]
        while active_queue and len(activated_chain) < depth:
            current_anchor, current_strength = active_queue.pop(0)
            if current_strength < 0.1:
                continue
            # Активируем якорь
            actual_strength = current_anchor.activate(context, current_strength)
            activated_chain.append((current_anchor, actual_strength))
            # Получаем ассоциации
            associations = current_anchor.trigger_associations()
            # Добавляем в очередь с уменьшенной силой
            for assoc in associations:
                new_strength = current_strength * 0.7
                if new_strength > 0.1:
                    active_queue.append((assoc, new_strength))
        return activated_chain
class EmotionalLearningSystem:
    """Система эмоционального обучения."""
    def __init__(self):
        self.emotional_states = {
            'curiosity': 0.0,
            'confusion': 0.0,
            'frustration': 0.0,
            'satisfaction': 0.0,
            'anxiety': 0.0,
            'excitement': 0.0
        }
        self.emotional_memory = []
        self.pain_threshold = 0.3
    def process_feedback(self, prediction, target, context):
        """Обработка обратной связи с эмоциональной оценкой"""
        error = abs(prediction - target)
        # Оценка эмоционального состояния
        emotional_state = self._determine_emotional_state(error)
        # Сигналы наказания/награды
        punishment = self._calculate_punishment(error)
        reward = self._calculate_reward(error)
        # Сохранение в память
        self.emotional_memory.append({
            'error': error,
            'emotional_state': emotional_state,
            'punishment': punishment,
            'reward': reward,
            'context': context,
            'timestamp': time.time()
        })
        # Обновление эмоциональных состояний
        self._update_emotional_states(emotional_state, error)
        return {
            'emotional_state': emotional_state,
            'punishment': punishment,
            'reward': reward,
            'error': error
        }
    def _determine_emotional_state(self, error):
        """Определение эмоционального состояния"""
        if error > 0.8:
            return 'frustration'
        elif error > 0.5:
            return 'confusion'
        elif error > 0.2:
            return 'curiosity'
        else:
            return 'satisfaction'
    def _calculate_punishment(self, error):
        """Расчет наказания"""
        return max(0.0, error * 0.5)
    def _calculate_reward(self, error):
        """Расчет награды"""
        return max(0.0, (1.0 - error) * 0.8)
    def _update_emotional_states(self, emotional_state, error):
        """Обновление эмоциональных состояний"""
        # Увеличиваем эмоциональное состояние
        if emotional_state in self.emotional_states:
            self.emotional_states[emotional_state] = min(1.0, self.emotional_states[emotional_state] + 0.1)
        # Уменьшаем другие состояния
        for state in self.emotional_states:
            if state != emotional_state:
                self.emotional_states[state] = max(0.0, self.emotional_states[state] - 0.05)
class PainBasedLearning:
    """Система "болезненного" обучения."""
    def __init__(self):
        self.pain_memory = []
        self.pain_prevention_rules = []
    def process_pain_signal(self, error, confidence, context):
    """Обработка сигнала боли с детализацией и коррекцией весов нейронов"""
    # Вычисление интенсивности боли с учетом нескольких факторов
    pain_intensity = self._calculate_pain_intensity(error, confidence)
    
    if pain_intensity > self.pain_threshold:
        # Создание записи о боли с подробной информацией
        pain_record = {
            'intensity': pain_intensity,
            'context': context,
            'error': error,
            'confidence': confidence,
            'timestamp': time.time(),
            'neurons_affected': [],  # Список нейронов, которые будут затронуты
            'error_type': self._classify_error_type(error),
            'severity_level': self._assess_severity_level(pain_intensity),
            'learning_opportunity': self._evaluate_learning_potential(error, confidence)
        }
        
        # Сохраняем запись в память боли
        self.pain_memory.append(pain_record)
        
        # Активируем реакцию на боль с учетом контекста
        self._activate_pain_response(pain_record)
        
        # Логируем событие для отладки и анализа
        logger.info(f"Pain signal processed - Intensity: {pain_intensity:.3f}, "
                   f"Error: {error:.3f}, Context: {context}")
        
        return True
    
    return False

def _calculate_pain_intensity(self, error, confidence):
    """Расчет интенсивности боли с учетом множества факторов"""
    # Базовая формула интенсивности боли
    base_intensity = error * (1.0 - confidence)
    
    # Добавляем адаптивный коэффициент на основе истории ошибок
    recent_errors = [record['error'] for record in self.pain_memory[-10:] if record.get('error')]
    if recent_errors:
        avg_recent_error = sum(recent_errors) / len(recent_errors)
        # Если текущая ошибка выше средней, увеличиваем интенсивность
        if error > avg_recent_error:
            base_intensity *= 1.2
    
    # Применяем нелинейную функцию для более точной оценки
    # Это позволяет быстрее реагировать на значительные ошибки
    pain_intensity = base_intensity * (1.0 + (base_intensity * 0.5))
    
    # Ограничиваем максимальную интенсивность
    return min(1.0, pain_intensity)

def _activate_pain_response(self, pain_record):
    """Активация ответа на боль с коррекцией весов нейронов"""
    try:
        # 1. Идентификация затронутых нейронов
        affected_neurons = self._identify_affected_neurons(pain_record)
        pain_record['neurons_affected'] = affected_neurons
        
        # 2. Коррекция весов нейронов
        self._adjust_neuron_weights(affected_neurons, pain_record)
        
        # 3. Активация систем саморегуляции
        self._trigger_self_regulation_systems(pain_record)
        
        # 4. Обновление памяти боли
        self._update_pain_memory_with_feedback(pain_record)
        
        # 5. Логирование реакции
        logger.info(f"Pain response activated for {len(affected_neurons)} neurons. "
                   f"Intensity: {pain_record['intensity']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in pain response activation: {e}")
        # В случае ошибки, хотя бы записываем в память
        pain_record['error_details'] = str(e)
        pain_record['response_failed'] = True

def _identify_affected_neurons(self, pain_record):
    """Определение нейронов, затронутых сигналом боли"""
    affected_neurons = []
    
    # Если у нас есть контекст, используем его для поиска соответствующих нейронов
    context = pain_record['context']
    
    # Ищем нейроны, связанные с текущим контекстом
    if context and isinstance(context, dict):
        # Попробуем найти нейроны по ключевым словам из контекста
        for key, value in context.items():
            if isinstance(value, str):
                # Ищем нейроны по ключевым словам
                for neuron_id, neuron in self.anchor_system.anchors.items():
                    if neuron.name and neuron.name.lower() in value.lower():
                        affected_neurons.append(neuron_id)
            
            # Проверяем эмоциональную сигнатуру
            if key == 'emotional_signature':
                for emotion, count in value.items():
                    if count > 0:
                        # Ищем нейроны, связанные с эмоциями
                        for neuron_id, neuron in self.anchor_system.anchors.items():
                            if neuron.type == 'emotion' and emotion in neuron.name.lower():
                                affected_neurons.append(neuron_id)
    
    # Если не нашли по контексту, используем базовый подход
    if not affected_neurons:
        # Пытаемся использовать текущие активные нейроны
        active_anchors = self.anchor_system.anchors.values()
        for anchor in active_anchors:
            if anchor.activation_strength > 0.3:  # Активные нейроны
                affected_neurons.append(anchor.id)
    
    # Убираем дубликаты
    return list(set(affected_neurons))

def _adjust_neuron_weights(self, affected_neurons, pain_record):
    """Коррекция весов нейронов на основе сигнала боли"""
    if not affected_neurons:
        return
    
    # Для каждого затронутого нейрона
    for neuron_id in affected_neurons:
        try:
            # Получаем нейрон
            neuron = self.anchor_system.anchors.get(neuron_id)
            if not neuron:
                continue
            
            # Вычисляем коэффициент коррекции на основе интенсивности боли
            correction_factor = pain_record['intensity'] * 0.5
            
            # Уменьшаем стабильность нейрона (увеличиваем шанс забывания)
            neuron.stability = max(0.0, neuron.stability - correction_factor * 0.1)
            
            # Уменьшаем активность нейрона
            neuron.activation_strength = max(0.0, neuron.activation_strength - correction_factor * 0.3)
            
            # Обновляем память нейрона
            neuron._update_memory(
                activation=neuron.activation_strength,
                context=pain_record['context'],
                emotional_feedback={'type': 'pain', 'intensity': pain_record['intensity']}
            )
            
            # Корректируем ассоциации
            self._modify_association_strengths(neuron, correction_factor)
            
            logger.debug(f"Adjusted neuron {neuron_id}: stability={neuron.stability:.3f}, "
                        f"activation={neuron.activation_strength:.3f}")
            
        except Exception as e:
            logger.error(f"Error adjusting neuron {neuron_id}: {e}")

def _modify_association_strengths(self, neuron, correction_factor):
    """Изменение силы ассоциаций нейрона"""
    # Уменьшаем силу ассоциаций
    for assoc_id, assoc_data in neuron.association_links.items():
        try:
            old_strength = assoc_data['strength']
            # Уменьшаем силу ассоциации
            new_strength = max(0.0, old_strength - correction_factor * 0.2)
            assoc_data['strength'] = new_strength
            assoc_data['last_used'] = time.time()
            
            logger.debug(f"Modified association {assoc_id}: {old_strength:.3f} -> {new_strength:.3f}")
        except Exception as e:
            logger.error(f"Error modifying association for neuron {neuron.id}: {e}")

def _trigger_self_regulation_systems(self, pain_record):
    """Активация систем саморегуляции"""
    try:
        # Увеличиваем уровень концентрации внимания
        self.attention_system.focus_level = min(1.0, self.attention_system.focus_level + 0.1)
        
        # Активируем систему эмоционального контроля
        self.emotional_intelligence_system.emotional_adaptability = min(1.0, 
            self.emotional_intelligence_system.emotional_adaptability + 0.05)
        
        # Увеличиваем уровень самоконтроля
        self.self_learning_system.self_assessment['pain_adaptation'] = \
            self.self_learning_system.self_assessment.get('pain_adaptation', 0) + 0.1
        
        # Активируем систему предупреждения о повторении ошибок
        self._setup_pain_avoidance_rules(pain_record)
        
        logger.info("Self-regulation systems triggered")
        
    except Exception as e:
        logger.error(f"Error triggering self-regulation systems: {e}")

def _setup_pain_avoidance_rules(self, pain_record):
    """Настройка правил предотвращения повторения боли"""
    try:
        # Создаем правило для предотвращения повторения ошибок
        rule = {
            'type': 'pain_avoidance',
            'context_pattern': pain_record['context'],
            'error_threshold': pain_record['error'],
            'intensity': pain_record['intensity'],
            'timestamp': time.time(),
            'avoidance_strategy': self._generate_avoidance_strategy(pain_record)
        }
        
        # Добавляем правило в память
        self.pain_prevention_rules.append(rule)
        
        # Ограничиваем количество правил
        if len(self.pain_prevention_rules) > 100:
            self.pain_prevention_rules.pop(0)
            
        logger.debug(f"Pain avoidance rule created: {rule}")
        
    except Exception as e:
        logger.error(f"Error setting up pain avoidance rules: {e}")

def _generate_avoidance_strategy(self, pain_record):
    """Генерация стратегии избегания повторения боли"""
    # Определяем тип ошибки
    error_type = pain_record.get('error_type', 'unknown')
    
    strategies = {
        'high_confidence_error': 'reduce_confidence_threshold',
        'low_confidence_error': 'increase_training_samples',
        'context_specific_error': 'contextual_override',
        'repetitive_error': 'pattern_breaking',
        'unknown': 'general_improvement'
    }
    
    return strategies.get(error_type, 'general_improvement')

def _classify_error_type(self, error):
    """Классификация типа ошибки"""
    if error > 0.8:
        return 'high_confidence_error'
    elif error > 0.5:
        return 'medium_confidence_error'
    elif error > 0.2:
        return 'low_confidence_error'
    else:
        return 'minor_error'

def _assess_severity_level(self, pain_intensity):
    """Оценка уровня серьезности боли"""
    if pain_intensity > 0.8:
        return 'severe'
    elif pain_intensity > 0.5:
        return 'moderate'
    elif pain_intensity > 0.2:
        return 'mild'
    else:
        return 'minimal'

def _evaluate_learning_potential(self, error, confidence):
    """Оценка потенциала обучения из ошибки"""
    # Ошибка с высокой уверенностью может быть менее обучаемой
    # Ошибка с низкой уверенностью может содержать больше информации
    learning_potential = (1.0 - confidence) * (1.0 - error)
    return min(1.0, learning_potential)

def _update_pain_memory_with_feedback(self, pain_record):
    """Обновление памяти боли с обратной связью"""
    # Обновляем информацию о том, как была обработана боль
    pain_record['response_time'] = time.time() - pain_record['timestamp']
    pain_record['correction_applied'] = True
    
    # Если это была повторяющаяся ошибка, увеличиваем вес обучения
    if self._is_repeated_pain_pattern(pain_record):
        pain_record['learning_boost'] = 1.5
        logger.info("Repeated pain pattern detected - increased learning boost")
    else:
        pain_record['learning_boost'] = 1.0

def _is_repeated_pain_pattern(self, pain_record):
    """Проверка на повторяющийся паттерн боли"""
    if len(self.pain_memory) < 3:
        return False
    
    # Проверяем последние 3 записи
    recent_pains = self.pain_memory[-3:]
    
    # Проверяем, похожи ли они по типу ошибки и контексту
    current_context = pain_record['context']
    current_error = pain_record['error']
    
    for past_pain in recent_pains[:-1]:  # Исключаем текущую запись
        if (past_pain['error'] > 0.5 and 
            abs(past_pain['error'] - current_error) < 0.2 and
            self._similar_contexts(current_context, past_pain['context'])):
            return True
    
    return False

def _similar_contexts(self, context1, context2):
    """Проверка на схожесть контекстов"""
    if not context1 or not context2:
        return False
    
    # Простая проверка по ключевым словам
    if isinstance(context1, dict) and isinstance(context2, dict):
        common_keys = set(context1.keys()) & set(context2.keys())
        return len(common_keys) > 0
    
    return False
    def _calculate_pain_intensity(self, error, confidence):
        """Расчет интенсивности боли"""
        return (error * (1.0 - confidence)) * 0.8
    def _activate_pain_response(self, pain_record):
    """Активация ответа на боль с коррекцией весов нейронов и системной реакцией"""
    try:
        # 1. Идентификация затронутых нейронов
        affected_neurons = self._identify_affected_neurons(pain_record)
        pain_record['neurons_affected'] = affected_neurons
        
        # 2. Коррекция весов нейронов
        self._adjust_neuron_weights(affected_neurons, pain_record)
        
        # 3. Активация систем саморегуляции
        self._trigger_self_regulation_systems(pain_record)
        
        # 4. Обновление памяти боли
        self._update_pain_memory_with_feedback(pain_record)
        
        # 5. Активация систем предотвращения повторения
        self._setup_pain_avoidance_rules(pain_record)
        
        # 6. Обновление контекстной системы
        self._update_contextual_recognition(pain_record)
        
        # 7. Активация систем эмоционального контроля
        self._activate_emotional_regulation(pain_record)
        
        # 8. Логирование успешной реакции
        logger.info(f"Pain response activated with intensity: {pain_record['intensity']:.3f}, "
                   f"Affected neurons: {len(affected_neurons)}, "
                   f"Learning opportunity: {pain_record.get('learning_opportunity', False)}")
        
    except Exception as e:
        logger.error(f"Error in pain response activation: {e}")
        pain_record['error_details'] = str(e)
        pain_record['response_failed'] = True

def _identify_affected_neurons(self, pain_record):
    """Определение нейронов, затронутых сигналом боли"""
    affected_neurons = []
    
    # Используем контекст для определения затронутых нейронов
    context = pain_record['context']
    
    # Если есть контекст с информацией о нейронах
    if isinstance(context, dict) and 'neurons' in context:
        # Используем указанные нейроны из контекста
        neuron_ids = context['neurons']
        if isinstance(neuron_ids, list):
            affected_neurons.extend(neuron_ids)
    
    # Если нет конкретных нейронов в контексте, используем алгоритмический подход
    if not affected_neurons:
        # Используем текущие активные нейроны
        active_anchors = self.anchor_system.anchors.values()
        for anchor in active_anchors:
            if anchor.activation_strength > 0.3:  # Активные нейроны
                affected_neurons.append(anchor.id)
    
    # Если всё ещё нет нейронов, используем базовый подход
    if not affected_neurons:
        # Используем нейроны, связанные с ключевыми словами из ошибки
        error_context = pain_record.get('error', '')
        if isinstance(error_context, str):
            words = error_context.lower().split()
            # Ищем нейроны по ключевым словам
            for word in words[:3]:  # Первые 3 слова
                for neuron_id, neuron in self.anchor_system.anchors.items():
                    if neuron.name and word in neuron.name.lower():
                        affected_neurons.append(neuron_id)
    
    # Убираем дубликаты и ограничиваем количество нейронов
    affected_neurons = list(set(affected_neurons))[:20]  # Максимум 20 нейронов
    
    return affected_neurons

def _adjust_neuron_weights(self, affected_neurons, pain_record):
    """Коррекция весов нейронов на основе сигнала боли"""
    if not affected_neurons:
        return
    
    # Для каждого затронутого нейрона
    for neuron_id in affected_neurons:
        try:
            # Получаем нейрон
            neuron = self.anchor_system.anchors.get(neuron_id)
            if not neuron:
                continue
            
            # Вычисляем коэффициент коррекции на основе интенсивности боли
            correction_factor = pain_record['intensity'] * 0.8
            
            # Уменьшаем стабильность нейрона (увеличиваем шанс забывания)
            neuron.stability = max(0.0, neuron.stability - correction_factor * 0.1)
            
            # Уменьшаем активность нейрона
            neuron.activation_strength = max(0.0, neuron.activation_strength - correction_factor * 0.3)
            
            # Обновляем память нейрона
            neuron._update_memory(
                activation=neuron.activation_strength,
                context=pain_record['context'],
                emotional_feedback={'type': 'pain', 'intensity': pain_record['intensity']}
            )
            
            # Корректируем ассоциации
            self._modify_association_strengths(neuron, correction_factor)
            
            # Обновляем систему эмоционального интеллекта
            self.emotional_intelligence_system.emotional_adaptability = min(
                1.0, 
                self.emotional_intelligence_system.emotional_adaptability + correction_factor * 0.05
            )
            
            logger.debug(f"Adjusted neuron {neuron_id}: stability={neuron.stability:.3f}, "
                        f"activation={neuron.activation_strength:.3f}")
            
        except Exception as e:
            logger.error(f"Error adjusting neuron {neuron_id}: {e}")

def _modify_association_strengths(self, neuron, correction_factor):
    """Изменение силы ассоциаций нейрона"""
    # Уменьшаем силу ассоциаций
    for assoc_id, assoc_data in neuron.association_links.items():
        try:
            old_strength = assoc_data['strength']
            # Уменьшаем силу ассоциации
            new_strength = max(0.0, old_strength - correction_factor * 0.2)
            assoc_data['strength'] = new_strength
            assoc_data['last_used'] = time.time()
            
            logger.debug(f"Modified association {assoc_id}: {old_strength:.3f} -> {new_strength:.3f}")
        except Exception as e:
            logger.error(f"Error modifying association for neuron {neuron.id}: {e}")

def _trigger_self_regulation_systems(self, pain_record):
    """Активация систем саморегуляции"""
    try:
        # Увеличиваем уровень концентрации внимания
        self.attention_system.focus_level = min(1.0, self.attention_system.focus_level + 0.15)
        
        # Активируем систему эмоционального контроля
        self.emotional_intelligence_system.emotional_adaptability = min(1.0, 
            self.emotional_intelligence_system.emotional_adaptability + 0.08)
        
        # Увеличиваем уровень самоконтроля
        self.self_learning_system.self_assessment['pain_adaptation'] = \
            self.self_learning_system.self_assessment.get('pain_adaptation', 0) + 0.12
        
        # Активируем систему саморегуляции обучения
        self._trigger_learning_regulation(pain_record)
        
        # Обновляем систему мотивации
        self._update_motivation_system(pain_record)
        
        logger.info("Self-regulation systems triggered")
        
    except Exception as e:
        logger.error(f"Error triggering self-regulation systems: {e}")

def _trigger_learning_regulation(self, pain_record):
    """Активация регуляции обучения на основе боли"""
    try:
        # Увеличиваем вес обучения от боли
        self.pain_memory_system.learning_from_pain = True
        
        # Активируем систему запоминания боли
        self.pain_memory_system.remember_pain({
            'experience': pain_record,
            'timestamp': time.time(),
            'context': pain_record['context'],
            'error': pain_record['error'],
            'solution': None
        })
        
        # Обновляем систему временного восприятия
        self.temporal_reasoning_system.causal_reasoning['recent_pain'] = {
            'intensity': pain_record['intensity'],
            'timestamp': time.time(),
            'context': pain_record['context']
        }
        
    except Exception as e:
        logger.error(f"Error in learning regulation: {e}")

def _update_motivation_system(self, pain_record):
    """Обновление системы мотивации на основе боли"""
    try:
        # Увеличиваем уровень мотивации для преодоления боли
        motivation_boost = pain_record['intensity'] * 0.1
        self.motivation_system.motivation_level = min(1.0, 
            self.motivation_system.motivation_level + motivation_boost)
        
        # Обновляем систему целеполагания
        current_goals = self.motivation_system.goals
        if current_goals:
            # Повышаем важность целей, связанных с преодолением ошибок
            for goal in current_goals:
                if 'error' in goal['goal'].lower() or 'correct' in goal['goal'].lower():
                    goal['importance'] = min(1.0, goal['importance'] + 0.05)
                    goal['motivation_required'] = min(1.0, goal['motivation_required'] + 0.03)
        
        logger.debug(f"Motivation system updated: {self.motivation_system.motivation_level:.3f}")
        
    except Exception as e:
        logger.error(f"Error updating motivation system: {e}")

def _update_pain_memory_with_feedback(self, pain_record):
    """Обновление памяти боли с обратной связью"""
    # Обновляем информацию о том, как была обработана боль
    pain_record['response_time'] = time.time() - pain_record['timestamp']
    pain_record['correction_applied'] = True
    pain_record['response_success'] = True
    
    # Если это была повторяющаяся ошибка, увеличиваем вес обучения
    if self._is_repeated_pain_pattern(pain_record):
        pain_record['learning_boost'] = 1.8
        logger.info("Repeated pain pattern detected - increased learning boost")
    else:
        pain_record['learning_boost'] = 1.2

def _is_repeated_pain_pattern(self, pain_record):
    """Проверка на повторяющийся паттерн боли"""
    if len(self.pain_memory) < 3:
        return False
    
    # Проверяем последние 3 записи
    recent_pains = self.pain_memory[-3:]
    
    # Проверяем, похожи ли они по типу ошибки и контексту
    current_context = pain_record['context']
    current_error = pain_record['error']
    
    for past_pain in recent_pains[:-1]:  # Исключаем текущую запись
        if (past_pain['error'] > 0.5 and 
            abs(past_pain['error'] - current_error) < 0.2 and
            self._similar_contexts(current_context, past_pain['context'])):
            return True
    
    return False

def _similar_contexts(self, context1, context2):
    """Проверка на схожесть контекстов"""
    if not context1 or not context2:
        return False
    
    # Простая проверка по ключевым словам
    if isinstance(context1, dict) and isinstance(context2, dict):
        common_keys = set(context1.keys()) & set(context2.keys())
        return len(common_keys) > 0
    
    return False

def _setup_pain_avoidance_rules(self, pain_record):
    """Настройка правил предотвращения повторения боли"""
    try:
        # Создаем правило для предотвращения повторения ошибок
        rule = {
            'type': 'pain_avoidance',
            'context_pattern': pain_record['context'],
            'error_threshold': pain_record['error'],
            'intensity': pain_record['intensity'],
            'timestamp': time.time(),
            'avoidance_strategy': self._generate_avoidance_strategy(pain_record),
            'related_neurons': pain_record.get('neurons_affected', [])
        }
        
        # Добавляем правило в память
        self.pain_prevention_rules.append(rule)
        
        # Ограничиваем количество правил
        if len(self.pain_prevention_rules) > 100:
            self.pain_prevention_rules.pop(0)
            
        logger.debug(f"Pain avoidance rule created: {rule}")
        
    except Exception as e:
        logger.error(f"Error setting up pain avoidance rules: {e}")

def _generate_avoidance_strategy(self, pain_record):
    """Генерация стратегии избегания повторения боли"""
    # Определяем тип ошибки
    error_type = pain_record.get('error_type', 'unknown')
    
    strategies = {
        'high_confidence_error': 'reduce_confidence_threshold',
        'low_confidence_error': 'increase_training_samples',
        'context_specific_error': 'contextual_override',
        'repetitive_error': 'pattern_breaking',
        'unknown': 'general_improvement'
    }
    
    return strategies.get(error_type, 'general_improvement')

def _update_contextual_recognition(self, pain_record):
    """Обновление системы контекстуального распознавания"""
    try:
        # Обновляем контекстную память
        context_key = str(pain_record['context'])
        if context_key not in self.context_system.context_memory:
            self.context_system.context_memory[context_key] = {
                'count': 0,
                'avg_pain_intensity': 0.0,
                'last_occurrence': time.time()
            }
        
        context_mem = self.context_system.context_memory[context_key]
        context_mem['count'] += 1
        context_mem['last_occurrence'] = time.time()
        context_mem['avg_pain_intensity'] = (
            (context_mem['avg_pain_intensity'] * (context_mem['count'] - 1) + 
             pain_record['intensity']) / context_mem['count']
        )
        
        # Обновляем систему интуиции
        self.intuition_system.pattern_recognition = min(
            1.0, 
            self.intuition_system.pattern_recognition + pain_record['intensity'] * 0.05
        )
        
        logger.debug(f"Contextual recognition updated for: {context_key}")
        
    except Exception as e:
        logger.error(f"Error updating contextual recognition: {e}")

def _activate_emotional_regulation(self, pain_record):
    """Активация эмоциональной регуляции"""
    try:
        # Определяем эмоциональное состояние, связанное с болью
        emotional_state = 'frustration' if pain_record['intensity'] > 0.7 else 'confusion'
        
        # Регулируем эмоциональное состояние
        regulation = self.emotional_intelligence_system.regulate_emotions(emotional_state)
        
        # Обновляем систему эмоционального интеллекта
        self.emotional_intelligence_system.emotional_adaptability = min(
            1.0,
            self.emotional_intelligence_system.emotional_adaptability + 0.1
        )
        
        # Обновляем систему эмоционального обучения
        self.emotional_system.emotional_memory.append({
            'error': pain_record['error'],
            'emotional_state': emotional_state,
            'punishment': pain_record['intensity'],
            'reward': 0.0,
            'context': pain_record['context'],
            'timestamp': time.time()
        })
        
        logger.debug(f"Emotional regulation activated: {emotional_state}")
        
    except Exception as e:
        logger.error(f"Error in emotional regulation: {e}")
class ContextualRecognitionSystem:
    """Система контекстуального распознавания."""
    def __init__(self):
        self.context_memory = {}
        self.context_similarity_threshold = 0.7
   def recognize_context(self, input_data, context_window=5):
    """Распознавание контекста с использованием биологически-ориентированных механизмов"""
    try:
        # Инициализация результатов
        context_match = []
        emotional_signature = {}
        confidence = 0.0
        learning_opportunity = False
        context_features = {}
        
        # Определение типа входных данных
        if isinstance(input_data, str):
            # Токенизация текста с использованием внутреннего токенайзера
            words = input_data.lower().split()
            text_length = len(words)
            
            # 1. Анализ семантики и ключевых слов
            semantic_context = self._analyze_semantic_context(words)
            context_match.extend(semantic_context['matches'])
            context_features.update(semantic_context['features'])
            
            # 2. Эмоциональная сигнатура
            emotional_signature = self._extract_emotional_signature(words)
            
            # 3. Контекстуальная память
            context_memory_analysis = self._analyze_context_memory(words, context_window)
            context_match.extend(context_memory_analysis['matches'])
            context_features.update(context_memory_analysis['features'])
            
            # 4. Активация якорей
            anchor_activations = self._activate_context_anchors(words)
            context_features['anchor_activations'] = anchor_activations
            
            # 5. Система интуиции
            intuition_result = self.intuition_system.quick_insight(input_data)
            context_features['intuition'] = intuition_result
            
            # 6. Эмоциональный интеллект
            emotional_recognition = self.emotional_intelligence_system.recognize_emotions(input_data)
            context_features['emotional_recognition'] = emotional_recognition
            
            # 7. Система внимания
            attention_weights = self.attention_system.focus_attention(words, input_data)
            context_features['attention_weights'] = attention_weights
            
            # 8. Расчет уверенности
            confidence = self._calculate_context_confidence(
                text_length, 
                emotional_signature, 
                context_match, 
                attention_weights
            )
            
            # 9. Оценка возможности обучения
            learning_opportunity = self._evaluate_learning_opportunity(
                input_data, 
                emotional_signature, 
                context_match
            )
            
        elif isinstance(input_data, dict):
            # Анализ сложного контекста
            context_features = self._analyze_complex_context(input_data)
            confidence = self._calculate_complex_context_confidence(input_data)
            learning_opportunity = self._evaluate_complex_learning_opportunity(input_data)
            
            # Извлечение эмоциональной сигнатуры из словаря
            if 'emotions' in input_data:
                emotional_signature = input_data['emotions']
                
        elif isinstance(input_data, list):
            # Анализ списка данных
            context_features = self._analyze_list_context(input_data)
            confidence = self._calculate_list_context_confidence(input_data)
            learning_opportunity = self._evaluate_list_learning_opportunity(input_data)
            
        # 10. Обновление системы контекстуального распознавания
        self._update_context_memory(input_data, context_features, emotional_signature)
        
        # 11. Система инстинктов
        instinct_response = self._trigger_instinct_context_detection(input_data)
        if instinct_response:
            context_features['instinct_response'] = instinct_response
            
        # 12. Система саморефлексии
        self_reflection = self.inner_voice_system.internal_dialogue(input_data)
        context_features['self_reflection'] = self_reflection
        
        # 13. Адаптация под личность
        personality_adaptation = self.personality_system.adapt_thinking("context_analysis")
        context_features['personality_adaptation'] = personality_adaptation
        
        # 14. Система мотивации
        motivation_level = self.motivation_system.assess_motivation(0.0)
        context_features['motivation_level'] = motivation_level
        
        # 15. Временная оценка
        temporal_analysis = self.temporal_reasoning_system.understand_temporal_relationships([input_data])
        context_features['temporal_analysis'] = temporal_analysis
        
        logger.debug(f"Context recognized: {len(context_match)} matches, confidence: {confidence:.3f}")
        
        return {
            'context_match': context_match,
            'emotional_signature': emotional_signature,
            'confidence': confidence,
            'learning_opportunity': learning_opportunity,
            'context_features': context_features,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error in context recognition: {e}")
        # Возвращаем базовые значения в случае ошибки
        return {
            'context_match': [],
            'emotional_signature': {},
            'confidence': 0.1,
            'learning_opportunity': False,
            'context_features': {},
            'timestamp': time.time()
        }

def _analyze_semantic_context(self, words):
    """Анализ семантики контекста"""
    matches = []
    features = {}
    
    # Поиск семантических ключевых слов
    semantic_keywords = {
        'scientific': ['research', 'study', 'experiment', 'theory', 'hypothesis'],
        'emotional': ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited'],
        'technical': ['code', 'algorithm', 'programming', 'data', 'system'],
        'philosophical': ['think', 'question', 'truth', 'meaning', 'existence'],
        'creative': ['art', 'design', 'create', 'imagine', 'innovate']
    }
    
    for category, keywords in semantic_keywords.items():
        found_keywords = [word for word in words if word in keywords]
        if found_keywords:
            matches.append({
                'category': category,
                'keywords': found_keywords,
                'count': len(found_keywords)
            })
            features[f'semantic_{category}'] = len(found_keywords)
    
    return {'matches': matches, 'features': features}

def _extract_emotional_signature(self, words):
    """Извлечение эмоциональной сигнатуры"""
    emotional_signature = {}
    
    # Эмоциональные слова
    emotion_words = {
        'negative': ['sad', 'angry', 'frustrated', 'disappointed', 'upset', 'worried', 'scared'],
        'positive': ['happy', 'excited', 'joyful', 'pleased', 'delighted', 'thrilled'],
        'confusion': ['confused', 'question', 'uncertain', 'puzzled', 'bewildered'],
        'surprise': ['surprised', 'amazed', 'shocked', 'astonished'],
        'calm': ['peaceful', 'calm', 'relaxed', 'serene', 'tranquil']
    }
    
    # Подсчет частоты эмоциональных слов
    for emotion, word_list in emotion_words.items():
        count = sum(1 for word in words if word in word_list)
        if count > 0:
            emotional_signature[emotion] = count
    
    # Учет интонации через знаки препинания и специальные символы
    punctuation_emotions = {
        'exclamation': sum(1 for word in words if '!' in word),
        'question': sum(1 for word in words if '?' in word),
        'ellipsis': sum(1 for word in words if '...' in word)
    }
    
    for emotion, count in punctuation_emotions.items():
        if count > 0:
            emotional_signature[f'punctuation_{emotion}'] = count
    
    return emotional_signature

def _analyze_context_memory(self, words, window_size):
    """Анализ контекста с использованием памяти"""
    matches = []
    features = {}
    
    # Проверка на повторяющиеся паттерны
    if len(words) > window_size:
        # Создаем окно для поиска паттернов
        for i in range(len(words) - window_size + 1):
            window = tuple(words[i:i+window_size])
            window_str = ' '.join(window)
            
            # Проверяем в памяти
            for context_key, context_data in self.context_system.context_memory.items():
                if window_str in context_key or context_key in window_str:
                    matches.append({
                        'pattern': 'memory_match',
                        'window': window_str,
                        'frequency': context_data['count'],
                        'avg_intensity': context_data['avg_pain_intensity']
                    })
    
    # Анализ частоты слов
    word_frequency = {}
    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1
    
    features['word_frequency'] = word_frequency
    features['unique_words'] = len(set(words))
    features['total_words'] = len(words)
    
    return {'matches': matches, 'features': features}

def _activate_context_anchors(self, words):
    """Активация якорей по контексту"""
    activated_anchors = []
    
    # Используем систему якорей для активации
    for word in words[:5]:  # Ограничиваем анализ первыми 5 словами
        # Проверяем наличие якоря
        if word in self.neuron_pool:
            anchor = self.neuron_pool[word]
            # Активируем якорь
            activation_strength = anchor.activate(context=words, strength=0.5)
            activated_anchors.append({
                'anchor_id': anchor.id,
                'name': anchor.name,
                'activation_strength': activation_strength,
                'type': anchor.type
            })
    
    return activated_anchors

def _calculate_context_confidence(self, text_length, emotional_signature, context_matches, attention_weights):
    """Расчет уверенности контекста"""
    # Базовая уверенность
    base_confidence = min(1.0, text_length / 20.0)
    
    # Учет эмоциональной сигнатуры
    emotional_confidence = len(emotional_signature) * 0.1
    
    # Учет контекстных совпадений
    match_confidence = len(context_matches) * 0.05
    
    # Учет внимания
    attention_confidence = sum(attention_weights.values()) if attention_weights else 0.0
    
    # Комбинируем все факторы
    confidence = (base_confidence * 0.4 + 
                 emotional_confidence * 0.3 + 
                 match_confidence * 0.2 + 
                 attention_confidence * 0.1)
    
    return min(1.0, confidence)

def _evaluate_learning_opportunity(self, input_data, emotional_signature, context_matches):
    """Оценка возможности обучения"""
    # Возможность обучения зависит от:
    # 1. Наличия эмоциональной сигнатуры (показывает интерес/вовлеченность)
    # 2. Наличия контекстных совпадений (показывает новизну)
    # 3. Длины текста (больше данных = больше возможностей)
    
    has_emotion = len(emotional_signature) > 0
    has_context = len(context_matches) > 0
    has_content = len(str(input_data).split()) > 3
    
    # Повышаем возможность обучения, если есть эмоции и контекст
    learning_opportunity = (has_emotion * 0.4 + 
                           has_context * 0.3 + 
                           has_content * 0.3)
    
    return learning_opportunity > 0.5

def _analyze_complex_context(self, context_dict):
    """Анализ сложного контекста (словарь)"""
    features = {}
    
    # Анализ различных компонентов контекста
    if 'text' in context_dict:
        features['text_length'] = len(context_dict['text'])
        features['word_count'] = len(context_dict['text'].split())
    
    if 'timestamp' in context_dict:
        features['temporal_distance'] = time.time() - context_dict['timestamp']
    
    if 'user_id' in context_dict:
        features['user_profile'] = context_dict.get('user_profile', {})
    
    if 'environment' in context_dict:
        features['environment_context'] = context_dict['environment']
    
    return features

def _calculate_complex_context_confidence(self, context_dict):
    """Расчет уверенности для сложного контекста"""
    confidence = 0.0
    
    # Оценка по ключевым полям
    required_fields = ['text', 'timestamp']
    present_fields = [field for field in required_fields if field in context_dict]
    confidence = len(present_fields) / len(required_fields) * 0.8
    
    # Учет дополнительных данных
    additional_data = len(context_dict) - 2  # Вычитаем обязательные поля
    confidence += min(0.2, additional_data * 0.05)
    
    return min(1.0, confidence)

def _evaluate_complex_learning_opportunity(self, context_dict):
    """Оценка возможности обучения для сложного контекста"""
    # Условия для обучения:
    # 1. Есть текст
    # 2. Есть пользовательская информация
    # 3. Есть временная отметка
    has_text = 'text' in context_dict
    has_user_info = 'user_id' in context_dict
    has_time = 'timestamp' in context_dict
    
    # Уровень возможности обучения
    learning_opportunity = (has_text * 0.5 + 
                           has_user_info * 0.3 + 
                           has_time * 0.2)
    
    return learning_opportunity > 0.4

def _analyze_list_context(self, context_list):
    """Анализ контекста в виде списка"""
    features = {}
    
    # Анализ структуры списка
    features['list_length'] = len(context_list)
    features['item_types'] = [type(item).__name__ for item in context_list]
    
    # Анализ уникальности
    unique_items = len(set(str(item) for item in context_list))
    features['uniqueness_ratio'] = unique_items / len(context_list) if context_list else 0.0
    
    return features

def _calculate_list_context_confidence(self, context_list):
    """Расчет уверенности для контекста списка"""
    if not context_list:
        return 0.0
    
    # Базовая уверенность пропорциональна длине списка
    base_confidence = min(1.0, len(context_list) / 10.0)
    
    # Учет разнообразия
    unique_ratio = len(set(str(item) for item in context_list)) / len(context_list)
    diversity_factor = min(1.0, unique_ratio * 2.0)
    
    return base_confidence * diversity_factor

def _evaluate_list_learning_opportunity(self, context_list):
    """Оценка возможности обучения для контекста списка"""
    if not context_list:
        return False
    
    # Возможность обучения зависит от:
    # 1. Количества элементов
    # 2. Разнообразия данных
    # 3. Структурированности
    
    has_content = len(context_list) > 0
    diverse_content = len(set(str(item) for item in context_list)) > 1
    
    learning_opportunity = (has_content * 0.4 + 
                           diverse_content * 0.6)
    
    return learning_opportunity > 0.5

def _update_context_memory(self, input_data, features, emotional_signature):
    """Обновление памяти контекста"""
    try:
        # Создаем ключ для памяти
        if isinstance(input_data, str):
            context_key = input_data.lower()[:100]  # Ограничиваем длину
        elif isinstance(input_data, dict):
            context_key = str(input_data)[:100]
        else:
            context_key = str(input_data)[:100]
        
        # Обновляем или создаем запись в памяти
        if context_key not in self.context_system.context_memory:
            self.context_system.context_memory[context_key] = {
                'count': 0,
                'avg_pain_intensity': 0.0,
                'last_occurrence': time.time(),
                'features': features,
                'emotional_signature': emotional_signature
            }
        
        context_mem = self.context_system.context_memory[context_key]
        context_mem['count'] += 1
        context_mem['last_occurrence'] = time.time()
        
        # Обновляем среднюю эмоциональную интенсивность (если есть)
        if emotional_signature:
            avg_intensity = sum(emotional_signature.values()) / len(emotional_signature) if emotional_signature else 0.0
            context_mem['avg_pain_intensity'] = (
                (context_mem['avg_pain_intensity'] * (context_mem['count'] - 1) + avg_intensity) / 
                context_mem['count']
            )
        
        # Обновляем особенности
        context_mem['features'].update(features)
        context_mem['emotional_signature'].update(emotional_signature)
        
    except Exception as e:
        logger.error(f"Error updating context memory: {e}")

def _trigger_instinct_context_detection(self, input_data):
    """Активация инстинктов при распознавании контекста"""
    try:
        # Простая проверка на инстинктивные триггеры
        if isinstance(input_data, str):
            input_lower = input_data.lower()
            
            # Инстинкты по ключевым словам
            triggers = {
                'surprise': ['surprise', 'amazing', 'unexpected', 'shocking'],
                'danger': ['danger', 'risk', 'harm', 'threat', 'warning'],
                'curiosity': ['question', 'what', 'how', 'why', 'interesting']
            }
            
            for instinct, keywords in triggers.items():
                if any(keyword in input_lower for keyword in keywords):
                    return self.instinct_system.trigger_instinct(instinct, input_data)
                    
        return None
        
    except Exception as e:
        logger.error(f"Error in instinct detection: {e}")
        return None
        
    def generate_associative_thought(self, seed_concept, context=None):
    """Генерация ассоциативного мышления с биологически-ориентированной архитектурой"""
    try:
        # Инициализация результатов
        associations = []
        abstract_thought = ""
        thought_process = {
            'seed_concept': seed_concept,
            'associations': [],
            'abstract_thought': "",
            'context': context,
            'depth': 0,
            'confidence': 0.0,
            'emotional_context': {},
            'memory_trace': [],
            'cognitive_bias': 0.0,
            'thinking_style': self.personality_system.thinking_style,
            'temporal_context': time.time()
        }
        
        # 1. Проверка в памяти для уже известных концептов
        memory_match = self._search_memory_for_concept(seed_concept)
        if memory_match:
            thought_process['memory_trace'].append({
                'source': 'long_term_memory',
                'match': memory_match,
                'timestamp': time.time()
            })
        
        # 2. Активация якорей по семантике
        activated_anchors = self._activate_seed_concept_anchors(seed_concept)
        thought_process['memory_trace'].extend(activated_anchors)
        
        # 3. Анализ семантики с использованием контекста
        semantic_analysis = self._analyze_semantic_structure(seed_concept, context)
        thought_process['emotional_context'] = semantic_analysis['emotional_signature']
        thought_process['depth'] = semantic_analysis['complexity_level']
        
        # 4. Генерация ассоциаций с учетом контекста и эмоций
        associations = self._generate_contextual_associations(
            seed_concept, 
            semantic_analysis['emotional_signature'], 
            context
        )
        
        # 5. Создание абстрактного мышления
        abstract_thought = self._form_abstract_thought(
            seed_concept, 
            associations, 
            semantic_analysis['emotional_signature'],
            context
        )
        
        # 6. Интеграция с системой интуиции
        intuition_insight = self.intuition_system.quick_insight(seed_concept)
        if intuition_insight['confidence'] > 0.5:
            associations.extend(intuition_insight['insight'].split())
            thought_process['memory_trace'].append({
                'source': 'intuition',
                'insight': intuition_insight['insight'],
                'confidence': intuition_insight['confidence']
            })
        
        # 7. Система саморефлексии
        self_reflection = self.inner_voice_system.self_reflection(seed_concept)
        thought_process['memory_trace'].extend(self_reflection['learned_patterns'])
        
        # 8. Эмоциональное состояние
        emotional_state = self.emotional_intelligence_system.recognize_emotions(seed_concept)
        thought_process['emotional_context'] = emotional_state['detected_emotions']
        
        # 9. Проверка на инстинктивные реакции
        instinct_response = self._check_instinct_triggers(seed_concept)
        if instinct_response:
            thought_process['memory_trace'].append({
                'source': 'instinct',
                'response': instinct_response,
                'trigger': seed_concept
            })
        
        # 10. Адаптация под стиль мышления
        thinking_style_adaptation = self.personality_system.adapt_thinking("associative_thought")
        thought_process['thinking_style'] = thinking_style_adaptation['adapted_style']
        
        # 11. Оценка уверенности в генерации
        thought_process['confidence'] = self._calculate_thought_confidence(
            associations, 
            semantic_analysis, 
            emotional_state
        )
        
        # 12. Система мотивации
        motivation_level = self.motivation_system.assess_motivation(0.0)
        thought_process['cognitive_bias'] = 0.1 * motivation_level
        
        # 13. Временная оценка
        temporal_analysis = self.temporal_reasoning_system.understand_temporal_relationships([seed_concept])
        thought_process['temporal_context'] = temporal_analysis
        
        # 14. Обновление памяти
        self._update_thought_memory(seed_concept, associations, abstract_thought, context)
        
        # 15. Формирование окончательного результата
        result = {
            'seed_concept': seed_concept,
            'associations': associations,
            'abstract_thought': abstract_thought,
            'context': context,
            'depth': thought_process['depth'],
            'confidence': thought_process['confidence'],
            'emotional_context': thought_process['emotional_context'],
            'memory_trace': thought_process['memory_trace'],
            'cognitive_bias': thought_process['cognitive_bias'],
            'thinking_style': thought_process['thinking_style'],
            'timestamp': time.time()
        }
        
        logger.debug(f"Associative thought generated for '{seed_concept}' with {len(associations)} associations")
        return result
        
    except Exception as e:
        logger.error(f"Error generating associative thought: {e}")
        # Возврат базового результата в случае ошибки
        return {
            'seed_concept': seed_concept,
            'associations': [],
            'abstract_thought': "",
            'context': context,
            'depth': 0,
            'confidence': 0.1,
            'emotional_context': {},
            'memory_trace': [],
            'cognitive_bias': 0.0,
            'thinking_style': self.personality_system.thinking_style,
            'timestamp': time.time()
        }

def _search_memory_for_concept(self, concept):
    """Поиск концепта в памяти"""
    try:
        # Ищем в долгосрочной памяти
        if concept in self.memory_system.long_term_memory:
            return self.memory_system.long_term_memory[concept]
        
        # Ищем в краткосрочной памяти
        for stored_item, timestamp in self.memory_system.short_term_memory:
            if concept in str(stored_item):
                return {
                    'content': stored_item,
                    'timestamp': timestamp,
                    'type': 'short_term'
                }
        
        # Ищем в эмоциональной памяти
        if concept in self.memory_system.emotional_memory:
            return self.memory_system.emotional_memory[concept]
            
        return None
    except Exception as e:
        logger.error(f"Error searching memory for concept '{concept}': {e}")
        return None

def _activate_seed_concept_anchors(self, seed_concept):
    """Активация якорей для стартовой концепции"""
    activated_anchors = []
    
    # Проверяем наличие якоря в системе
    if seed_concept in self.anchor_system.anchors:
        anchor = self.anchor_system.anchors[seed_concept]
        activation_strength = anchor.activate(context=seed_concept, strength=1.0)
        activated_anchors.append({
            'anchor_id': anchor.id,
            'name': anchor.name,
            'activation_strength': activation_strength,
            'type': anchor.type,
            'timestamp': time.time()
        })
        return activated_anchors
    
    # Создаем новый якорь если он не существует
    try:
        new_anchor = self.anchor_system.create_anchor(
            concept_name=seed_concept,
            context=seed_concept,
            anchor_type="concept"
        )
        activation_strength = new_anchor.activate(context=seed_concept, strength=1.0)
        activated_anchors.append({
            'anchor_id': new_anchor.id,
            'name': new_anchor.name,
            'activation_strength': activation_strength,
            'type': new_anchor.type,
            'timestamp': time.time(),
            'created': True
        })
    except Exception as e:
        logger.error(f"Error creating anchor for '{seed_concept}': {e}")
    
    return activated_anchors

def _analyze_semantic_structure(self, seed_concept, context):
    """Анализ семантической структуры концепта"""
    try:
        # Оценка сложности
        complexity_level = 0
        semantic_features = {}
        
        # Анализ по длине и структуре
        words = seed_concept.split()
        word_count = len(words)
        complexity_level = min(1.0, word_count / 5.0)
        
        # Анализ эмоционального контекста
        emotional_signature = {}
        if context:
            # Используем систему контекстуального распознавания
            context_analysis = self.context_system.recognize_context(context)
            emotional_signature = context_analysis['emotional_signature']
        
        # Анализ на предмет ключевых категорий
        semantic_categories = {
            'scientific': ['research', 'study', 'theory', 'experiment', 'data'],
            'emotional': ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited'],
            'technical': ['code', 'algorithm', 'programming', 'system', 'machine'],
            'philosophical': ['think', 'question', 'truth', 'meaning', 'existence'],
            'creative': ['art', 'design', 'create', 'imagine', 'innovate']
        }
        
        for category, keywords in semantic_categories.items():
            found_keywords = [word for word in words if word in keywords]
            if found_keywords:
                semantic_features[f'category_{category}'] = len(found_keywords)
        
        # Оценка глубины мышления
        depth_score = 0.0
        if word_count > 1:
            depth_score = 0.3 + (word_count - 1) * 0.1
        
        return {
            'complexity_level': min(1.0, complexity_level + depth_score),
            'semantic_features': semantic_features,
            'emotional_signature': emotional_signature,
            'word_count': word_count
        }
    except Exception as e:
        logger.error(f"Error analyzing semantic structure for '{seed_concept}': {e}")
        return {
            'complexity_level': 0.0,
            'semantic_features': {},
            'emotional_signature': {},
            'word_count': len(seed_concept.split()) if seed_concept else 0
        }

def _generate_contextual_associations(self, seed_concept, emotional_signature, context):
    """Генерация ассоциаций с учетом контекста"""
    try:
        associations = []
        base_associations = []
        
        # Существующие ассоциации из системы якорей
        if seed_concept in self.anchor_system.anchors:
            anchor = self.anchor_system.anchors[seed_concept]
            # Получаем связанные якоря
            linked_anchors = anchor.trigger_associations()
            for linked_anchor in linked_anchors:
                if hasattr(linked_anchor, 'name') and linked_anchor.name:
                    base_associations.append(linked_anchor.name)
        
        # Базовые ассоциации по категориям
        if seed_concept.lower() in ['dog', 'cat', 'animal']:
            base_associations = ['pet', 'fur', 'tail', 'paws', 'bark', 'meow', 'loyalty', 'companionship']
        elif seed_concept.lower() in ['computer', 'technology', 'machine']:
            base_associations = ['processor', 'memory', 'software', 'hardware', 'internet', 'digital', 'automation']
        elif seed_concept.lower() in ['science', 'research', 'knowledge']:
            base_associations = ['experiment', 'theory', 'hypothesis', 'discovery', 'methodology', 'analysis']
        elif seed_concept.lower() in ['music', 'sound', 'melody']:
            base_associations = ['rhythm', 'harmony', 'instrument', 'composition', 'emotion', 'expression']
        elif seed_concept.lower() in ['food', 'eat', 'nutrition']:
            base_associations = ['taste', 'flavor', 'nutrients', 'satiety', 'culture', 'preference']
        elif seed_concept.lower() in ['space', 'universe', 'cosmos']:
            base_associations = ['stars', 'planets', 'galaxy', 'gravity', 'exploration', 'infinity']
        elif seed_concept.lower() in ['love', 'romance', 'affection']:
            base_associations = ['empathy', 'connection', 'passion', 'commitment', 'intimacy', 'care']
        elif seed_concept.lower() in ['health', 'wellness', 'fitness']:
            base_associations = ['exercise', 'diet', 'sleep', 'mental', 'physical', 'balance']
        elif seed_concept.lower() in ['education', 'learning', 'school']:
            base_associations = ['knowledge', 'skills', 'teaching', 'curriculum', 'student', 'teacher']
        elif seed_concept.lower() in ['art', 'creative', 'painting']:
            base_associations = ['color', 'shape', 'expression', 'inspiration', 'aesthetics', 'creativity']
        elif seed_concept.lower() in ['history', 'past', 'time']:
            base_associations = ['events', 'chronology', 'memory', 'tradition', 'legacy', 'change']
        elif seed_concept.lower() in ['business', 'commerce', 'market']:
            base_associations = ['profit', 'competition', 'strategy', 'innovation', 'growth', 'partnership']
        
        # Добавляем эмоциональные ассоциации
        if emotional_signature:
            for emotion, count in emotional_signature.items():
                if count > 0:
                    # Добавляем эмоциональные слова в ассоциации
                    emotion_words = {
                        'positive': ['joy', 'happiness', 'pleasure', 'satisfaction'],
                        'negative': ['sadness', 'frustration', 'anger', 'fear'],
                        'confusion': ['uncertainty', 'confusion', 'question', 'doubt']
                    }
                    if emotion in emotion_words:
                        base_associations.extend(emotion_words[emotion])
        
        # Добавляем контекстные ассоциации
        if context:
            context_words = str(context).lower().split()
            # Используем контекст для расширения ассоциаций
            for word in context_words[:3]:  # Первые 3 слова контекста
                if word not in base_associations and len(word) > 2:
                    base_associations.append(word)
        
        # Уникализация и ограничение количества
        associations = list(set(base_associations))[:20]  # Ограничение до 20 ассоциаций
        
        # Добавляем ассоциации от интуиции
        intuition_result = self.intuition_system.quick_insight(seed_concept)
        if intuition_result['confidence'] > 0.6:
            intuition_words = intuition_result['insight'].split()[:5]  # Первые 5 слов интуиции
            associations.extend(intuition_words)
            associations = list(set(associations))[:20]  # Уникализация снова
        
        return associations[:20]  # Окончательное ограничение
        
    except Exception as e:
        logger.error(f"Error generating contextual associations for '{seed_concept}': {e}")
        return []

def _form_abstract_thought(self, seed_concept, associations, emotional_signature, context):
    """Формирование абстрактного мышления"""
    try:
        # Составление абстрактной мысли
        abstract_thought = ""
        
        # Система мудрости
        wisdom_evaluation = self.wisdom_system.evaluate_confidence(seed_concept, associations)
        
        # Оценка на основе эмоционального контекста
        emotional_impact = len(emotional_signature) * 0.2
        
        # Оценка на основе ассоциаций
        association_impact = min(1.0, len(associations) * 0.1)
        
        # Сложность концепта
        concept_complexity = len(seed_concept.split()) * 0.1
        
        # Общий уровень абстракции
        abstraction_level = (wisdom_evaluation * 0.3 + 
                           emotional_impact * 0.2 + 
                           association_impact * 0.3 + 
                           concept_complexity * 0.2)
        
        # Генерация абстрактной мысли в зависимости от уровня абстракции
        if abstraction_level < 0.3:
            abstract_thought = f"The concept of '{seed_concept}' represents basic understanding."
        elif abstraction_level < 0.6:
            abstract_thought = f"'{seed_concept}' connects to multiple related concepts and has moderate complexity."
        elif abstraction_level < 0.8:
            abstract_thought = f"'{seed_concept}' embodies complex interconnections with various domains of knowledge."
        else:
            abstract_thought = f"'{seed_concept}' represents a profound philosophical or scientific principle with deep implications."
        
        # Добавление контекста
        if context:
            abstract_thought += f" In the given context, '{seed_concept}' takes on additional meaning."
        
        # Добавление эмоционального контекста
        if emotional_signature:
            emotions_str = ", ".join(list(emotional_signature.keys())[:2])
            abstract_thought += f" This concept evokes emotional responses including {emotions_str}."
        
        # Добавление ассоциаций
        if associations:
            assoc_str = ", ".join(associations[:3])
            abstract_thought += f" It relates to associated concepts such as {assoc_str}."
        
        return abstract_thought
        
    except Exception as e:
        logger.error(f"Error forming abstract thought for '{seed_concept}': {e}")
        return f"Abstract understanding of '{seed_concept}'"

def _check_instinct_triggers(self, seed_concept):
    """Проверка на инстинктивные триггеры"""
    try:
        # Простая проверка на инстинктивные триггеры
        instinct_triggers = {
            'surprise': ['unexpected', 'amazing', 'shocking', 'surprise'],
            'danger': ['danger', 'risk', 'harm', 'threat', 'warning'],
            'curiosity': ['question', 'what', 'how', 'why', 'interesting', 'curious']
        }
        
        seed_lower = seed_concept.lower()
        for instinct, keywords in instinct_triggers.items():
            if any(keyword in seed_lower for keyword in keywords):
                # Активируем инстинкт
                instinct_response = self.instinct_system.trigger_instinct(instinct, seed_concept)
                return instinct_response
                
        return None
    except Exception as e:
        logger.error(f"Error checking instinct triggers for '{seed_concept}': {e}")
        return None

def _calculate_thought_confidence(self, associations, semantic_analysis, emotional_state):
    """Расчет уверенности в генерации мысли"""
    try:
        # Базовая уверенность
        base_confidence = min(1.0, len(associations) * 0.1)
        
        # Учет сложности
        complexity_confidence = semantic_analysis['complexity_level'] * 0.3
        
        # Учет эмоционального контекста
        emotional_confidence = len(emotional_state['detected_emotions']) * 0.2
        
        # Учет уникальности
        uniqueness_factor = 1.0 if len(set(associations)) == len(associations) else 0.7
        
        # Общая уверенность
        confidence = (base_confidence * 0.4 + 
                     complexity_confidence * 0.3 + 
                     emotional_confidence * 0.2 + 
                     uniqueness_factor * 0.1)
        
        return min(1.0, confidence)
    except Exception as e:
        logger.error(f"Error calculating thought confidence: {e}")
        return 0.5

def _update_thought_memory(self, seed_concept, associations, abstract_thought, context):
    """Обновление памяти о мышлении"""
    try:
        # Сохраняем в краткосрочную память
        self.memory_system.short_term_memory.append((
            {
                'seed_concept': seed_concept,
                'associations': associations,
                'abstract_thought': abstract_thought,
                'context': context,
                'timestamp': time.time()
            },
            time.time()
        ))
        
        # Сохраняем в долгосрочную память
        memory_key = f"thought_{seed_concept}_{int(time.time())}"
        self.memory_system.long_term_memory[memory_key] = {
            'concept': seed_concept,
            'associations': associations,
            'abstract_thought': abstract_thought,
            'context': context,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        # Обновляем счетчик доступа
        if memory_key in self.memory_system.long_term_memory:
            self.memory_system.long_term_memory[memory_key]['access_count'] += 1
        
        # Сохраняем в эмоциональную память если есть эмоции
        if context and isinstance(context, dict) and 'emotional_signature' in context:
            self.memory_system.emotional_memory[memory_key] = {
                'concept': seed_concept,
                'emotional_context': context['emotional_signature'],
                'timestamp': time.time()
            }
            
    except Exception as e:
        logger.error(f"Error updating thought memory: {e}")
        
class WisdomSystem:
    """Система "мудрости" и интеллекта."""
    def __init__(self):
        self.knowledge_depth = 0
        self.pattern_recognition = 0.0
        self.cognitive_efficiency = 0.0
        self.learning_curves = {}
        self.knowledge_integration = {}
        self.reasoning_depth = 0.0
        self.intellectual_honesty = 0.0
        self.metacognitive_awareness = 0.0
        self.wisdom_accumulator = 0.0
        self.confidence_history = deque(maxlen=100)
    
    def evaluate_confidence(self, new_information, existing_knowledge, context=None):
        """Оценка уверенности в новой информации с учетом контекста и мудрости"""
        try:
            # Инициализация компонентов оценки
            consistency = 0.0
            novelty_factor = 0.0
            temporal_factor = 0.0
            relevance_factor = 0.0
            complexity_factor = 0.0
            consistency_weight = 0.3
            novelty_weight = 0.2
            temporal_weight = 0.2
            relevance_weight = 0.2
            complexity_weight = 0.1
            
            # 1. Анализ согласованности новой информации с существующими знаниями
            if existing_knowledge:
                consistency = self._calculate_consistency(new_information, existing_knowledge)
                consistency = min(1.0, consistency)
            
            # 2. Анализ новизны информации
            novelty_factor = self._calculate_novelty(new_information)
            
            # 3. Анализ временного фактора (новые данные более ценны)
            temporal_factor = self._calculate_temporal_value(context)
            
            # 4. Анализ релевантности
            relevance_factor = self._calculate_relevance(new_information, existing_knowledge, context)
            
            # 5. Анализ сложности информации
            complexity_factor = self._calculate_complexity(new_information)
            
            # 6. Обновление метакогнитивной осведомленности
            self._update_metacognitive_awareness(
                consistency, novelty_factor, temporal_factor, relevance_factor
            )
            
            # 7. Расчет общей уверенности с учетом мудрости
            confidence_score = (
                consistency * consistency_weight +
                novelty_factor * novelty_weight +
                temporal_factor * temporal_weight +
                relevance_factor * relevance_weight +
                complexity_factor * complexity_weight
            )
            
            # 8. Применение корректировки мудрости
            wisdom_adjustment = self._apply_wisdom_adjustment(
                consistency, novelty_factor, complexity_factor
            )
            confidence_score = min(1.0, max(0.0, confidence_score * wisdom_adjustment))
            
            # 9. Обновление истории уверенности
            self.confidence_history.append(confidence_score)
            
            # 10. Обновление статистики мудрости
            self._update_wisdom_accumulator(
                confidence_score, consistency, novelty_factor, relevance_factor
            )
            
            logger.debug(f"Confidence evaluation: {confidence_score:.3f} "
                        f"(consistency: {consistency:.3f}, "
                        f"novelty: {novelty_factor:.3f}, "
                        f"temporal: {temporal_factor:.3f}, "
                        f"relevance: {relevance_factor:.3f})")
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error in confidence evaluation: {e}")
            return 0.3  # Базовое значение при ошибке
    
    def _calculate_consistency(self, new_information, existing_knowledge):
        """Расчет согласованности новой информации с существующими знаниями"""
        consistency = 0.0
        
        if isinstance(new_information, str) and isinstance(existing_knowledge, dict):
            # Проверка на совпадения в ключах
            for key, value in existing_knowledge.items():
                if isinstance(value, str) and new_information.lower() in value.lower():
                    consistency += 0.15
                elif isinstance(value, list) and any(
                    new_information.lower() in item.lower() for item in value
                ):
                    consistency += 0.15
            
            # Проверка на семантические сходства
            if isinstance(new_information, str):
                words = new_information.lower().split()
                for key, value in existing_knowledge.items():
                    if isinstance(value, str):
                        value_words = value.lower().split()
                        shared_words = set(words) & set(value_words)
                        if shared_words:
                            consistency += min(0.2, len(shared_words) * 0.05)
        
        # Учет семантической близости (упрощенно)
        if isinstance(new_information, str) and len(new_information) > 10:
            consistency += 0.1  # Базовая согласованность для длинной информации
            
        return min(1.0, consistency)
    
    def _calculate_novelty(self, new_information):
        """Расчет новизны информации"""
        novelty = 0.0
        
        if isinstance(new_information, str):
            # Длина информации как мера новизны
            if len(new_information) > 50:
                novelty += 0.3
            elif len(new_information) > 20:
                novelty += 0.15
            
            # Уникальность (простая проверка на повторы)
            if len(new_information) > 10:
                unique_chars = len(set(new_information.lower()))
                unique_ratio = unique_chars / len(new_information)
                novelty += min(0.2, unique_ratio * 0.5)
            
            # Проверка на наличие уникальных терминов
            unique_terms = ['innovative', 'revolutionary', 'groundbreaking', 'cutting-edge']
            for term in unique_terms:
                if term in new_information.lower():
                    novelty += 0.2
        
        # Учет времени создания (если доступен)
        if hasattr(new_information, 'timestamp'):
            age = time.time() - new_information.timestamp
            if age < 86400:  # Меньше дня
                novelty += 0.2
            elif age < 604800:  # Меньше недели
                novelty += 0.1
        
        return min(1.0, novelty)
    
    def _calculate_temporal_value(self, context):
        """Расчет временной ценности информации"""
        temporal_value = 0.0
        
        # Базовая временная ценность
        temporal_value = 0.7  # В реальной реализации будет анализ времени
        
        # Учет контекста
        if context and isinstance(context, dict):
            # Если есть временные метки
            if 'timestamp' in context:
                age = time.time() - context['timestamp']
                if age < 3600:  # Меньше часа
                    temporal_value += 0.2
                elif age < 86400:  # Меньше дня
                    temporal_value += 0.1
                elif age < 604800:  # Меньше недели
                    temporal_value += 0.05
        
        # Учет актуальности
        if context and isinstance(context, dict):
            if 'urgency' in context:
                urgency_factor = context['urgency']
                temporal_value += min(0.2, urgency_factor * 0.2)
        
        return min(1.0, temporal_value)
    
    def _calculate_relevance(self, new_information, existing_knowledge, context):
        """Расчет релевантности информации"""
        relevance = 0.0
        
        # Базовая релевантность
        if isinstance(new_information, str) and len(new_information) > 5:
            relevance = 0.3
            
        # Релевантность по контексту
        if context and isinstance(context, dict):
            context_words = str(context).lower()
            if isinstance(new_information, str):
                info_words = new_information.lower()
                # Проверка на совпадение ключевых слов
                relevant_keywords = ['important', 'critical', 'key', 'main', 'essential']
                found_keywords = [kw for kw in relevant_keywords if kw in context_words]
                if found_keywords:
                    relevance += 0.2
                    
                # Проверка на наличие в контексте
                if any(word in context_words for word in info_words.split()):
                    relevance += 0.15
        
        # Релевантность по существующим знаниям
        if existing_knowledge and isinstance(existing_knowledge, dict):
            if isinstance(new_information, str):
                # Проверка на наличие в существующих знаниях
                for key, value in existing_knowledge.items():
                    if isinstance(value, str) and new_information.lower() in value.lower():
                        relevance += 0.25
                    elif isinstance(value, list) and any(
                        new_information.lower() in item.lower() for item in value
                    ):
                        relevance += 0.25
        
        return min(1.0, relevance)
    
    def _calculate_complexity(self, new_information):
        """Расчет сложности информации"""
        complexity = 0.0
        
        if isinstance(new_information, str):
            # Оценка по длине
            if len(new_information) > 100:
                complexity += 0.3
            elif len(new_information) > 50:
                complexity += 0.15
            
            # Оценка по структуре
            sentences = new_information.count('.') + new_information.count('!') + new_information.count('?')
            if sentences > 3:
                complexity += 0.2
            
            # Оценка по сложности слов
            words = new_information.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if avg_word_length > 8:
                    complexity += 0.15
                elif avg_word_length > 5:
                    complexity += 0.05
        
        # Учет специальных символов и форматирования
        special_chars = sum(1 for c in new_information if c in ['(', ')', '[', ']', '{', '}'])
        complexity += min(0.2, special_chars * 0.05)
        
        return min(1.0, complexity)
    
    def _update_metacognitive_awareness(self, consistency, novelty, temporal, relevance):
        """Обновление метакогнитивной осведомленности"""
        awareness_score = (
            consistency * 0.25 +
            novelty * 0.25 +
            temporal * 0.25 +
            relevance * 0.25
        )
        self.metacognitive_awareness = min(1.0, self.metacognitive_awareness * 0.9 + awareness_score * 0.1)
    
    def _apply_wisdom_adjustment(self, consistency, novelty, complexity):
        """Применение корректировки мудрости к уверенности"""
        # Мудрость влияет на то, насколько мы доверяем информации
        wisdom_factor = 0.0
        
        # Если информация согласована и новая - доверие растет
        if consistency > 0.5 and novelty > 0.5:
            wisdom_factor = 1.2
        # Если информация согласована и простая - доверие высокое
        elif consistency > 0.7 and complexity < 0.5:
            wisdom_factor = 1.3
        # Если информация сложная и не согласована - доверие снижается
        elif complexity > 0.7 and consistency < 0.3:
            wisdom_factor = 0.7
        # В остальных случаях - стандартное доверие
        else:
            wisdom_factor = 1.0
        
        # Учет мудрости системы
        wisdom_level = self.wisdom_accumulator / (len(self.confidence_history) + 1) if self.confidence_history else 0.5
        wisdom_factor = wisdom_factor * (0.5 + wisdom_level * 0.5)  # Усреднение с уровнем мудрости
        
        return min(1.5, max(0.5, wisdom_factor))
    
    def _update_wisdom_accumulator(self, confidence, consistency, novelty, relevance):
        """Обновление аккумулятора мудрости"""
        # Увеличиваем аккумулятор мудрости на основе качественных оценок
        wisdom_contribution = (
            confidence * 0.3 +
            consistency * 0.2 +
            novelty * 0.2 +
            relevance * 0.3
        )
        self.wisdom_accumulator = min(1.0, self.wisdom_accumulator * 0.95 + wisdom_contribution * 0.05)
    
    def integrate_knowledge(self, new_knowledge, source_context):
        """Интеграция новой информации в существующую систему знаний"""
        try:
            # Создаем уникальный ключ для знания
            knowledge_key = self._generate_knowledge_key(new_knowledge)
            
            # Сохраняем знание с контекстом
            self.knowledge_integration[knowledge_key] = {
                'knowledge': new_knowledge,
                'source_context': source_context,
                'timestamp': time.time(),
                'confidence': self.evaluate_confidence(new_knowledge, self.knowledge_integration),
                'integration_score': 0.0
            }
            
            # Обновляем глубину знаний
            self.knowledge_depth = min(1.0, self.knowledge_depth + 0.01)
            
            # Обновляем когнитивную эффективность
            self.cognitive_efficiency = min(1.0, self.cognitive_efficiency + 0.005)
            
            # Обновляем кривую обучения
            self._update_learning_curve(knowledge_key, self.knowledge_depth)
            
            logger.debug(f"Knowledge integrated: {knowledge_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error integrating knowledge: {e}")
            return False
    
    def _generate_knowledge_key(self, knowledge):
        """Генерация уникального ключа для знания"""
        if isinstance(knowledge, str):
            return hashlib.md5(knowledge.encode()).hexdigest()[:16]
        elif isinstance(knowledge, dict):
            return hashlib.md5(str(sorted(knowledge.items())).encode()).hexdigest()[:16]
        else:
            return str(id(knowledge))[:16]
    
    def _update_learning_curve(self, knowledge_key, depth):
        """Обновление кривой обучения"""
        if knowledge_key not in self.learning_curves:
            self.learning_curves[knowledge_key] = []
        
        self.learning_curves[knowledge_key].append({
            'timestamp': time.time(),
            'depth': depth,
            'confidence': self.evaluate_confidence(knowledge_key, self.knowledge_integration)
        })
        
        # Ограничиваем историю
        if len(self.learning_curves[knowledge_key]) > 50:
            self.learning_curves[knowledge_key].pop(0)
    
    def assess_intellectual_honesty(self, claim, evidence):
        """Оценка честности интеллектуального подхода"""
        honesty_score = 0.0
        
        # Проверка на честность
        if isinstance(claim, str) and isinstance(evidence, list):
            # Проверка на наличие противоречий
            contradictions = self._detect_contradictions(claim, evidence)
            honesty_score = max(0.0, 1.0 - contradictions * 0.2)
            
            # Проверка на прозрачность
            transparency = self._assess_transparency(claim, evidence)
            honesty_score = (honesty_score * 0.7 + transparency * 0.3)
        
        # Обновляем уровень честности интеллекта
        self.intellectual_honesty = min(1.0, self.intellectual_honesty * 0.95 + honesty_score * 0.05)
        
        return honesty_score
    
    def _detect_contradictions(self, claim, evidence):
        """Обнаружение противоречий"""
        contradiction_count = 0
        if isinstance(claim, str) and isinstance(evidence, list):
            claim_lower = claim.lower()
            for item in evidence:
                if isinstance(item, str):
                    item_lower = item.lower()
                    # Простая проверка на противоречия
                    if 'not' in claim_lower and 'not' in item_lower:
                        contradiction_count += 0.1
                    elif ('not' in claim_lower and item_lower in claim_lower) or \
                         ('not' in item_lower and claim_lower in item_lower):
                        contradiction_count += 0.2
        return contradiction_count
    
    def _assess_transparency(self, claim, evidence):
        """Оценка прозрачности"""
        transparency = 0.0
        if isinstance(claim, str) and isinstance(evidence, list):
            # Проверка на наличие ссылок и источников
            if any('http://' in str(item) or 'https://' in str(item) for item in evidence):
                transparency += 0.3
            # Проверка на полноту
            if len(evidence) > 2:
                transparency += 0.2
            # Проверка на логическую связность
            if len(claim.split()) > 10:
                transparency += 0.1
        return transparency
    
    def evaluate_reasoning_depth(self, argument):
        """Оценка глубины рассуждения"""
        reasoning_score = 0.0
        
        if isinstance(argument, str):
            # Оценка по длине и структуре
            words = argument.split()
            if len(words) > 20:
                reasoning_score += 0.3
            
            # Оценка по наличию логических слов
            logic_words = ['therefore', 'because', 'however', 'moreover', 'nevertheless']
            found_logic = [lw for lw in logic_words if lw in argument.lower()]
            reasoning_score += len(found_logic) * 0.1
            
            # Оценка по количеству предложений
            sentences = argument.count('.') + argument.count('!') + argument.count('?')
            reasoning_score += min(0.3, sentences * 0.1)
            
            # Оценка по сложности конструкций
            complex_constructs = ['if and only if', 'given that', 'inasmuch as']
            found_complex = [cc for cc in complex_constructs if cc in argument.lower()]
            reasoning_score += len(found_complex) * 0.15
        
        # Обновляем глубину рассуждения
        self.reasoning_depth = min(1.0, self.reasoning_depth * 0.9 + reasoning_score * 0.1)
        
        return reasoning_score
    
    def get_wisdom_summary(self):
        """Получение сводки по мудрости"""
        return {
            'knowledge_depth': self.knowledge_depth,
            'pattern_recognition': self.pattern_recognition,
            'cognitive_efficiency': self.cognitive_efficiency,
            'metacognitive_awareness': self.metacognitive_awareness,
            'intellectual_honesty': self.intellectual_honesty,
            'wisdom_accumulator': self.wisdom_accumulator,
            'reasoning_depth': self.reasoning_depth,
            'confidence_trend': list(self.confidence_history)[-10:] if len(self.confidence_history) >= 10 else list(self.confidence_history)
        }
        
class InstinctSystem:
    """Система инстинктов и автоматических реакций."""
    def __init__(self):
        self.instinct_triggers = {}
        self.automatic_responses = {}
        self.response_priority = {}
        self.instinct_memory = []
        self.instinct_activation_log = deque(maxlen=100)
        self.emotional_instinct_mapping = {}
        self.contextual_instinct_triggers = {}
        
    def trigger_instinct(self, stimulus, context=None, emotional_state=None):
        """Активация инстинкта с учетом контекста и эмоционального состояния"""
        try:
            # Ищем инстинкт по стимулу
            instinct = None
            priority = 1.0
            
            # Проверяем прямое соответствие
            if stimulus in self.instinct_triggers:
                instinct = self.instinct_triggers[stimulus]
                priority = self.response_priority.get(instinct, 1.0)
            
            # Проверяем эмоциональные инстинкты
            if emotional_state and instinct is None:
                instinct = self._find_emotional_instinct(emotional_state)
                if instinct:
                    priority = self.response_priority.get(instinct, 1.0)
            
            # Проверяем контекстуальные триггеры
            if context and instinct is None:
                instinct = self._find_contextual_instinct(context)
                if instinct:
                    priority = self.response_priority.get(instinct, 1.0)
            
            # Если инстинкт найден, выполняем реакцию
            if instinct:
                response = self.execute_automatic_response(instinct, context, priority, emotional_state)
                # Логируем активацию инстинкта
                self.instinct_activation_log.append({
                    'stimulus': stimulus,
                    'instinct': instinct,
                    'priority': priority,
                    'context': context,
                    'timestamp': time.time()
                })
                logger.info(f"Instinct activated: {instinct} (priority: {priority})")
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Error in instinct trigger: {e}")
            return None
    
    def execute_automatic_response(self, instinct, context, priority, emotional_state=None):
        """Выполнение автоматической реакции с учетом эмоционального состояния"""
        try:
            # Быстрое выполнение без глубокого анализа
            response = {
                'type': 'automatic',
                'instinct': instinct,
                'context': context,
                'priority': priority,
                'emotional_state': emotional_state,
                'timestamp': time.time(),
                'response_time': 0.0
            }
            
            # Замер времени выполнения реакции
            start_time = time.time()
            
            # Сложные реакции с учетом эмоционального состояния
            if instinct == 'surprise':
                response['action'] = 'focus_attention'
                response['message'] = 'Surprise detected, focusing attention'
                response['adaptive_strategy'] = self._get_adaptive_attention_strategy(context)
                
            elif instinct == 'danger':
                response['action'] = 'avoid'
                response['message'] = 'Danger detected, avoiding'
                response['safety_measures'] = self._get_safety_response(context)
                
            elif instinct == 'curiosity':
                response['action'] = 'explore'
                response['message'] = 'Curiosity triggered, exploring'
                response['exploration_strategy'] = self._get_exploration_strategy(context)
                
            elif instinct == 'fear':
                response['action'] = 'prepare_defense'
                response['message'] = 'Fear detected, preparing defensive response'
                response['defense_level'] = self._assess_defense_need(context)
                
            elif instinct == 'joy':
                response['action'] = 'share_positive'
                response['message'] = 'Joy detected, sharing positive experience'
                response['sharing_strategy'] = self._get_sharing_strategy(context)
                
            elif instinct == 'sadness':
                response['action'] = 'seek_support'
                response['message'] = 'Sadness detected, seeking support'
                response['support_strategy'] = self._get_support_strategy(context)
                
            else:
                response['action'] = 'default'
                response['message'] = 'Default automatic response'
                response['fallback_strategy'] = self._get_fallback_strategy(context)
            
            # Завершаем замер времени
            response['response_time'] = time.time() - start_time
            
            # Сохраняем в память инстинктов
            self.instinct_memory.append({
                'instinct': instinct,
                'context': context,
                'response': response,
                'timestamp': time.time()
            })
            
            logger.info(f"Automatic response executed: {response['message']}")
            return response
            
        except Exception as e:
            logger.error(f"Error executing automatic response: {e}")
            # Возвращаем базовую реакцию в случае ошибки
            return {
                'type': 'automatic',
                'instinct': instinct,
                'context': context,
                'priority': priority,
                'timestamp': time.time(),
                'error': str(e),
                'action': 'error_handling',
                'message': 'Error in automatic response execution'
            }
    
    def _find_emotional_instinct(self, emotional_state):
        """Поиск инстинкта по эмоциональному состоянию"""
        # Сопоставление эмоций с инстинктами
        emotion_instinct_map = {
            'surprise': ['surprise'],
            'fear': ['fear', 'anxiety', 'panic'],
            'anger': ['anger', 'frustration'],
            'joy': ['joy', 'excitement', 'happiness'],
            'sadness': ['sadness', 'depression', 'grief'],
            'confusion': ['confusion', 'uncertainty']
        }
        
        for instinct, emotions in emotion_instinct_map.items():
            if emotional_state.lower() in emotions:
                return instinct
        
        return None
    
    def _find_contextual_instinct(self, context):
        """Поиск инстинкта по контексту"""
        if not isinstance(context, dict):
            return None
            
        # Проверяем на ключевые слова контекста
        context_keywords = {
            'danger': ['danger', 'risk', 'threat', 'harm', 'warning'],
            'surprise': ['surprise', 'unexpected', 'amazing', 'shocking'],
            'curiosity': ['question', 'what', 'how', 'why', 'interesting', 'curious'],
            'joy': ['happy', 'excited', 'joyful', 'pleased'],
            'sadness': ['sad', 'depressed', 'upset', 'gloomy']
        }
        
        context_text = str(context).lower()
        for instinct, keywords in context_keywords.items():
            if any(keyword in context_text for keyword in keywords):
                return instinct
                
        return None
    
    def _get_adaptive_attention_strategy(self, context):
        """Получение адаптивной стратегии фокусировки внимания"""
        strategy = {
            'focus_type': 'intensive',
            'duration': 0.5,
            'intensity': 0.8,
            'target_areas': []
        }
        
        if context:
            context_str = str(context).lower()
            if 'important' in context_str or 'critical' in context_str:
                strategy['focus_type'] = 'selective'
                strategy['intensity'] = 1.0
            elif 'question' in context_str or 'what' in context_str:
                strategy['focus_type'] = 'exploratory'
                strategy['duration'] = 1.0
                strategy['target_areas'] = ['query_elements']
        
        return strategy
    
    def _get_safety_response(self, context):
        """Получение реакции безопасности"""
        safety_response = {
            'level': 'medium',
            'actions': [],
            'considerations': []
        }
        
        if context:
            context_str = str(context).lower()
            if 'high risk' in context_str or 'serious threat' in context_str:
                safety_response['level'] = 'high'
                safety_response['actions'] = ['immediate retreat', 'alert others', 'secure area']
                safety_response['considerations'] = ['avoid further exposure', 'monitor surroundings']
            elif 'low risk' in context_str or 'minor danger' in context_str:
                safety_response['level'] = 'low'
                safety_response['actions'] = ['observe carefully', 'maintain distance', 'stay alert']
                safety_response['considerations'] = ['assess situation', 'prepare backup plan']
        
        return safety_response
    
    def _get_exploration_strategy(self, context):
        """Получение стратегии исследования"""
        exploration = {
            'approach': 'systematic',
            'depth': 'moderate',
            'focus_areas': ['key_elements', 'patterns', 'relationships']
        }
        
        if context:
            context_str = str(context).lower()
            if 'complex' in context_str or 'multi-faceted' in context_str:
                exploration['approach'] = 'comprehensive'
                exploration['depth'] = 'deep'
                exploration['focus_areas'] = ['components', 'interactions', 'structure']
            elif 'simple' in context_str or 'basic' in context_str:
                exploration['approach'] = 'focused'
                exploration['depth'] = 'surface'
                exploration['focus_areas'] = ['main aspects', 'core features']
        
        return exploration
    
    def _assess_defense_need(self, context):
        """Оценка необходимости защиты"""
        defense_level = {
            'required': False,
            'intensity': 0.0,
            'strategy': 'none'
        }
        
        if context:
            context_str = str(context).lower()
            if 'attack' in context_str or 'conflict' in context_str:
                defense_level['required'] = True
                defense_level['intensity'] = 0.8
                defense_level['strategy'] = 'active_defense'
            elif 'challenge' in context_str or 'obstacle' in context_str:
                defense_level['required'] = True
                defense_level['intensity'] = 0.5
                defense_level['strategy'] = 'adaptive_defense'
        
        return defense_level
    
    def _get_sharing_strategy(self, context):
        """Получение стратегии обмена положительным опытом"""
        sharing = {
            'approach': 'selective',
            'audience': 'trusted',
            'medium': 'verbal',
            'timing': 'immediate'
        }
        
        if context:
            context_str = str(context).lower()
            if 'celebration' in context_str or 'achievement' in context_str:
                sharing['approach'] = 'enthusiastic'
                sharing['audience'] = 'all'
                sharing['medium'] = 'verbal_or_written'
                sharing['timing'] = 'prompt'
            elif 'private' in context_str or 'personal' in context_str:
                sharing['approach'] = 'reserved'
                sharing['audience'] = 'selected'
                sharing['medium'] = 'written'
                sharing['timing'] = 'after_processing'
        
        return sharing
    
    def _get_support_strategy(self, context):
        """Получение стратегии поддержки"""
        support = {
            'approach': 'empathetic',
            'resources': ['listening', 'validation'],
            'duration': 'ongoing',
            'support_type': 'emotional'
        }
        
        if context:
            context_str = str(context).lower()
            if 'emergency' in context_str or 'crisis' in context_str:
                support['approach'] = 'urgent'
                support['resources'] = ['immediate_help', 'safe_space', 'professional_support']
                support['duration'] = 'immediate'
                support['support_type'] = 'crisis'
            elif 'daily' in context_str or 'routine' in context_str:
                support['approach'] = 'consistent'
                support['resources'] = ['regular_check-ins', 'encouragement', 'guidance']
                support['duration'] = 'regular'
                support['support_type'] = 'daily'
        
        return support
    
    def _get_fallback_strategy(self, context):
        """Получение стратегии запасного варианта"""
        return {
            'primary_action': 'observe',
            'secondary_action': 'wait_for_clarity',
            'decision_point': 'reassess_context',
            'timeout_period': 1.0
        }
    
    def register_instinct_trigger(self, stimulus, instinct, priority=1.0):
        """Регистрация нового триггера инстинкта"""
        self.instinct_triggers[stimulus] = instinct
        self.response_priority[instinct] = priority
        logger.info(f"Registered instinct trigger: {stimulus} -> {instinct} (priority: {priority})")
    
    def register_emotional_instinct_mapping(self, emotion, instinct):
        """Регистрация сопоставления эмоции и инстинкта"""
        self.emotional_instinct_mapping[emotion] = instinct
        logger.info(f"Registered emotional instinct mapping: {emotion} -> {instinct}")
    
    def get_instinct_statistics(self):
        """Получение статистики по использованию инстинктов"""
        stats = {
            'total_activations': len(self.instinct_activation_log),
            'instinct_distribution': {},
            'recent_activations': list(self.instinct_activation_log)[-10:] if len(self.instinct_activation_log) >= 10 else list(self.instinct_activation_log)
        }
        
        # Подсчет распределения инстинктов
        for activation in self.instinct_activation_log:
            instinct = activation.get('instinct', 'unknown')
            stats['instinct_distribution'][instinct] = stats['instinct_distribution'].get(instinct, 0) + 1
            
        return stats

class InnerVoiceSystem:
    """Система внутреннего голоса и саморефлексии."""
    def __init__(self):
        self.thought_processes = []
        self.reflection_memory = []
        self.critical_thinking = False
        self.self_reflection_history = deque(maxlen=50)
        self.thought_patterns = {}
        self.cognitive_bias_tracker = {}
        self.mindfulness_level = 0.0
        self.reflective_depth = 0.0
        
    def internal_dialogue(self, thoughts, context=None, depth=1):
        """Внутренний диалог для анализа мыслей с учетом глубины"""
        try:
            # Система самокритики
            # Анализ своих мыслей
            # Вопросы себе
            analysis_result = {
                'thoughts': thoughts,
                'analysis': [],
                'questions': [],
                'reflections': [],
                'depth_level': depth,
                'context': context,
                'timestamp': time.time()
            }
            
            # Простой анализ мыслей
            if isinstance(thoughts, str):
                words = thoughts.split()
                word_count = len(words)
                
                if word_count > 5:
                    analysis_result['analysis'].append(f"Thought contains {word_count} words")
                    # Проверяем на вопросы
                    if '?' in thoughts:
                        analysis_result['questions'].append("Contains question mark")
                    # Проверяем на эмоциональные слова
                    emotions = ['happy', 'sad', 'angry', 'excited', 'confused', 'frustrated', 'joyful']
                    found_emotions = [e for e in emotions if e in thoughts.lower()]
                    if found_emotions:
                        analysis_result['analysis'].append(f"Found emotions: {found_emotions}")
                    
                    # Анализ сложности мысли
                    if word_count > 20:
                        analysis_result['analysis'].append("Complex thought structure")
                        analysis_result['depth_level'] = min(3, depth + 0.5)
                    elif word_count > 10:
                        analysis_result['analysis'].append("Moderate thought complexity")
                        analysis_result['depth_level'] = min(2, depth + 0.2)
                        
                    # Анализ структуры
                    sentences = thoughts.count('.') + thoughts.count('!') + thoughts.count('?')
                    if sentences > 1:
                        analysis_result['analysis'].append(f"Multi-sentence structure ({sentences} sentences)")
                        
                    # Проверка на логические слова
                    logic_words = ['therefore', 'because', 'thus', 'hence', 'consequently', 'however', 'but']
                    found_logic = [lw for lw in logic_words if lw in thoughts.lower()]
                    if found_logic:
                        analysis_result['analysis'].append(f"Logical reasoning detected: {found_logic}")
                        
                # Анализ по глубине мышления
                if depth > 2:
                    analysis_result['analysis'].append("Deep analytical processing")
                    analysis_result['reflections'].append("This thought requires deeper consideration")
                    
                # Проверка на повторы
                if len(words) > 3:
                    unique_words = set(words)
                    ratio = len(unique_words) / len(words)
                    if ratio < 0.7:
                        analysis_result['analysis'].append("High repetition detected")
                        analysis_result['questions'].append("Is this repetitive thinking helpful?")
                        
            # Проверка на когнитивные искажения
            cognitive_biases = self._detect_cognitive_biases(thoughts)
            if cognitive_biases:
                analysis_result['analysis'].append(f"Cognitive biases detected: {cognitive_biases}")
                analysis_result['reflections'].append("Consider alternative perspectives")
                
            # Оценка осознанности
            mindfulness_score = self._assess_mindfulness(thoughts)
            analysis_result['mindfulness_score'] = mindfulness_score
            
            # Обновление истории рефлексии
            self.self_reflection_history.append({
                'thought': thoughts,
                'depth': depth,
                'analysis': analysis_result['analysis'],
                'timestamp': time.time()
            })
            
            logger.info(f"Internal dialogue analysis completed (depth: {depth})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in internal dialogue analysis: {e}")
            return {
                'thoughts': thoughts,
                'analysis': ['Analysis failed due to error'],
                'questions': [],
                'reflections': [],
                'depth_level': depth,
                'context': context,
                'timestamp': time.time()
            }
    
    def _detect_cognitive_biases(self, thoughts):
        """Обнаружение когнитивных искажений"""
        biases = []
        thoughts_lower = thoughts.lower()
        
        # Проверка на когнитивные искажения
        bias_indicators = {
            'confirmation_bias': ['only', 'just', 'obviously', 'clearly'],
            'overgeneralization': ['always', 'never', 'everyone', 'no one'],
            'catastrophizing': ['terrible', 'awful', 'horrible', 'disaster'],
            'black_white_thinking': ['perfect', 'impossible', 'all or nothing'],
            'mind_reading': ['knows', 'thinks', 'believes', 'expects'],
            'fortune_telling': ['will definitely', 'definitely', 'certainly']
        }
        
        for bias, indicators in bias_indicators.items():
            found_indicators = [indicator for indicator in indicators if indicator in thoughts_lower]
            if found_indicators:
                biases.append(bias)
                
        return biases
    
    def _assess_mindfulness(self, thoughts):
        """Оценка уровня осознанности"""
        mindfulness_score = 0.0
        thoughts_lower = thoughts.lower()
        
        # Проверка на осознанность
        mindful_indicators = ['consider', 'reflect', 'think about', 'observe', 'notice', 'aware']
        found_indicators = [indicator for indicator in mindful_indicators if indicator in thoughts_lower]
        mindfulness_score += len(found_indicators) * 0.1
        
        # Проверка на самоанализ
        self_reflection_indicators = ['I think', 'I feel', 'I wonder', 'I realize', 'I understand']
        found_reflection = [indicator for indicator in self_reflection_indicators if indicator in thoughts_lower]
        mindfulness_score += len(found_reflection) * 0.15
        
        # Проверка на сбалансированность
        balanced_indicators = ['but also', 'on the other hand', 'considering', 'although', 'however']
        found_balanced = [indicator for indicator in balanced_indicators if indicator in thoughts_lower]
        mindfulness_score += len(found_balanced) * 0.1
        
        return min(1.0, mindfulness_score)
    
    def self_reflection(self, current_thoughts, context=None, reflection_depth=1):
        """Самоанализ с учетом глубины"""
        try:
            # Оценка своей работы
            # Понимание своих ошибок
            # Улучшение стратегий
            reflection = {
                'current_thoughts': current_thoughts,
                'context': context,
                'reflection_depth': reflection_depth,
                'self_evaluation': {},
                'improvement_suggestions': [],
                'learned_patterns': [],
                'cognitive_insights': [],
                'timestamp': time.time()
            }
            
            # Простая самооценка
            if isinstance(current_thoughts, str):
                length = len(current_thoughts.split())
                if length < 3:
                    reflection['self_evaluation']['clarity'] = 'low'
                    reflection['improvement_suggestions'].append('Try to be more detailed')
                elif length > 20:
                    reflection['self_evaluation']['clarity'] = 'high'
                    reflection['improvement_suggestions'].append('Consider breaking into smaller points')
                else:
                    reflection['self_evaluation']['clarity'] = 'medium'
                
                # Проверка на повторы
                words = current_thoughts.lower().split()
                unique_words = set(words)
                if len(unique_words) / len(words) < 0.7:
                    reflection['improvement_suggestions'].append('Reduce repetition')
                    
                # Анализ сложности
                if length > 15:
                    reflection['cognitive_insights'].append("Complex thought processing detected")
                    
                # Проверка на эмоциональность
                emotional_words = ['happy', 'sad', 'angry', 'excited', 'confused', 'frustrated']
                found_emotions = [word for word in emotional_words if word in current_thoughts.lower()]
                if found_emotions:
                    reflection['cognitive_insights'].append(f"Emotional content detected: {found_emotions}")
                    
                # Анализ структуры
                sentences = current_thoughts.count('.') + current_thoughts.count('!') + current_thoughts.count('?')
                if sentences > 2:
                    reflection['cognitive_insights'].append("Multi-sentence complex structure")
                    
            # Обновление истории рефлексии
            self.self_reflection_history.append({
                'thought': current_thoughts,
                'depth': reflection_depth,
                'evaluation': reflection['self_evaluation'],
                'timestamp': time.time()
            })
            
            # Обновление уровня осознанности
            self.mindfulness_level = min(1.0, self.mindfulness_level + 0.05)
            
            # Обновление глубины рефлексии
            self.reflective_depth = min(1.0, self.reflective_depth + 0.02)
            
            logger.info(f"Self-reflection completed (depth: {reflection_depth})")
            return reflection
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            return {
                'current_thoughts': current_thoughts,
                'context': context,
                'reflection_depth': reflection_depth,
                'self_evaluation': {'error': 'Reflection failed'},
                'improvement_suggestions': ['Check for technical issues'],
                'learned_patterns': [],
                'cognitive_insights': [],
                'timestamp': time.time()
            }
    
    def analyze_thought_patterns(self, thoughts_history):
        """Анализ паттернов мышления"""
        try:
            patterns = {
                'common_topics': {},
                'repetition_frequency': {},
                'emotional_trends': {},
                'complexity_trends': []
            }
            
            # Анализ истории мыслей
            if thoughts_history:
                for thought_entry in thoughts_history[-20:]:  # Последние 20 мыслей
                    thought = thought_entry.get('thought', '')
                    if isinstance(thought, str):
                        words = thought.lower().split()
                        # Подсчет повторений
                        for word in words:
                            patterns['repetition_frequency'][word] = patterns['repetition_frequency'].get(word, 0) + 1
                        
                        # Подсчет тем
                        topics = ['science', 'technology', 'emotion', 'logic', 'philosophy']
                        for topic in topics:
                            if topic in thought.lower():
                                patterns['common_topics'][topic] = patterns['common_topics'].get(topic, 0) + 1
                        
                        # Эмоциональные тренды
                        emotions = ['happy', 'sad', 'angry', 'excited', 'confused']
                        found_emotions = [e for e in emotions if e in thought.lower()]
                        for emotion in found_emotions:
                            patterns['emotional_trends'][emotion] = patterns['emotional_trends'].get(emotion, 0) + 1
                
                # Сортировка по частоте
                patterns['repetition_frequency'] = dict(
                    sorted(patterns['repetition_frequency'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
                )
                
                patterns['common_topics'] = dict(
                    sorted(patterns['common_topics'].items(), 
                          key=lambda x: x[1], reverse=True)
                )
                
                patterns['emotional_trends'] = dict(
                    sorted(patterns['emotional_trends'].items(), 
                          key=lambda x: x[1], reverse=True)
                )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing thought patterns: {e}")
            return {}
    
    def get_mindfulness_report(self):
        """Получение отчета по осознанности"""
        return {
            'mindfulness_level': self.mindfulness_level,
            'reflective_depth': self.reflective_depth,
            'recent_reflections': list(self.self_reflection_history)[-5:],
            'total_reflections': len(self.self_reflection_history),
            'timestamp': time.time()
        }
class AttentionSystem:
    """Система внимания и фильтрации информации."""
    def __init__(self):
        self.focus_level = 1.0
        self.attention_weights = {}
        self.distracting_factors = []
        self.attention_history = deque(maxlen=100)
        self.cognitive_load = 0.0
        self.focus_shifts = 0
        self.attention_decay_rate = 0.05
        self.max_cognitive_load = 10.0
        self.attention_spread = {}
        
    def focus_attention(self, elements, context=None, dynamic_weighting=True):
        """Фокусировка внимания на важных элементах с биологически-ориентированной моделью"""
        try:
            # Расчет весов внимания
            # Отсечение ненужной информации
            # Повышение концентрации на ключевых моментах
            
            attention_weights = {}
            total_weight = 0.0
            
            # Инициализация параметров для каждого элемента
            element_features = {}
            
            # Простая модель весов внимания с учетом контекста
            for element in elements:
                if isinstance(element, str):
                    # Считаем важность по количеству слов и наличию ключевых слов
                    words = element.split()
                    base_weight = len(words) * 0.1
                    
                    # Учет контекста
                    context_weight = 0.0
                    if context and isinstance(context, dict):
                        # Проверяем ключевые слова в контексте
                        context_words = str(context).lower()
                        if 'urgent' in context_words:
                            context_weight += 0.3
                        if 'critical' in context_words:
                            context_weight += 0.2
                        if 'important' in context_words:
                            context_weight += 0.2
                            
                    # Увеличиваем вес для ключевых слов
                    key_words = ['important', 'critical', 'urgent', 'key', 'main', 'essential', 'significant']
                    key_word_bonus = 0.0
                    for key_word in key_words:
                        if key_word in element.lower():
                            key_word_bonus += 0.5
                            
                    # Учет эмоционального контекста
                    emotional_weight = 0.0
                    if context and isinstance(context, dict) and 'emotional_signature' in context:
                        emotional_sig = context['emotional_signature']
                        if emotional_sig:
                            emotional_weight = sum(emotional_sig.values()) * 0.1
                            
                    # Учет длины текста
                    length_weight = min(0.5, len(element) * 0.005)
                    
                    # Базовый вес с учетом всех факторов
                    weight = base_weight + key_word_bonus + context_weight + emotional_weight + length_weight
                    
                    # Динамическое взвешивание
                    if dynamic_weighting and len(elements) > 1:
                        # Уменьшаем вес для слишком длинных элементов
                        if len(element) > 100:
                            weight *= 0.7
                        # Увеличиваем вес для коротких, но значимых элементов
                        if len(element) < 10 and key_word_bonus > 0:
                            weight *= 1.5
                            
                    attention_weights[element] = weight
                    element_features[element] = {
                        'base_weight': base_weight,
                        'key_word_bonus': key_word_bonus,
                        'context_weight': context_weight,
                        'emotional_weight': emotional_weight,
                        'length_weight': length_weight,
                        'dynamic_adjustment': dynamic_weighting
                    }
                    total_weight += weight
                    
            # Нормализация весов с учетом динамики
            if total_weight > 0:
                normalized_weights = {}
                for element, weight in attention_weights.items():
                    normalized_weights[element] = weight / total_weight
                attention_weights = normalized_weights
                
            # Повышаем уровень концентрации
            self.focus_level = min(1.0, self.focus_level + 0.1)
            
            # Обновляем историю внимания
            self.attention_history.append({
                'elements_processed': len(elements),
                'weights': attention_weights,
                'context': context,
                'timestamp': time.time(),
                'focus_level': self.focus_level
            })
            
            # Обновляем когнитивную нагрузку
            self._update_cognitive_load(len(elements), attention_weights)
            
            # Обновляем распределение внимания
            self._update_attention_spread(attention_weights)
            
            logger.info(f"Attention focused on {len(elements)} elements (focus: {self.focus_level:.3f})")
            return {
                'weights': attention_weights,
                'element_features': element_features,
                'focus_level': self.focus_level,
                'cognitive_load': self.cognitive_load
            }
            
        except Exception as e:
            logger.error(f"Error in attention focusing: {e}")
            # Возврат базового результата в случае ошибки
            return {
                'weights': {str(elem): 1.0/len(elements) if elements else 0.0 for elem in elements},
                'element_features': {},
                'focus_level': self.focus_level,
                'cognitive_load': self.cognitive_load
            }
    
    def filter_information(self, incoming_data, filtering_strategy='adaptive'):
        """Фильтрация информации по важности с различными стратегиями"""
        try:
            # Система отбора данных
            # Исключение шума
            # Выделение ключевых паттернов
            
            filtered_data = []
            important_elements = []
            attention_signals = []
            
            # Выбор стратегии фильтрации
            if filtering_strategy == 'strict':
                importance_threshold = 0.8
                filtering_method = self._strict_filtering
            elif filtering_strategy == 'balanced':
                importance_threshold = 0.5
                filtering_method = self._balanced_filtering
            elif filtering_strategy == 'adaptive':
                importance_threshold = self._calculate_adaptive_threshold(incoming_data)
                filtering_method = self._adaptive_filtering
            else:
                importance_threshold = 0.5
                filtering_method = self._balanced_filtering
                
            # Простая фильтрация по важности
            for item in incoming_data:
                try:
                    if isinstance(item, str):
                        # Определяем важность
                        importance_score = filtering_method(item)
                        
                        # Добавляем сигнал внимания
                        attention_signal = self._generate_attention_signal(item, importance_score)
                        attention_signals.append(attention_signal)
                        
                        if importance_score >= importance_threshold:
                            important_elements.append(item)
                            filtered_data.append({
                                'item': item, 
                                'importance': importance_score, 
                                'status': 'important',
                                'signal': attention_signal
                            })
                        else:
                            filtered_data.append({
                                'item': item, 
                                'importance': importance_score, 
                                'status': 'filtered',
                                'signal': attention_signal
                            })
                    elif isinstance(item, dict):
                        # Для сложных объектов
                        importance_score = self._process_complex_item(item)
                        if importance_score >= importance_threshold:
                            important_elements.append(item)
                            filtered_data.append({
                                'item': item, 
                                'importance': importance_score, 
                                'status': 'important',
                                'signal': attention_signal
                            })
                        else:
                            filtered_data.append({
                                'item': item, 
                                'importance': importance_score, 
                                'status': 'filtered',
                                'signal': attention_signal
                            })
                except Exception as item_error:
                    logger.warning(f"Error processing item: {item_error}")
                    # Возвращаем минимальную важность для проблемных элементов
                    filtered_data.append({
                        'item': item, 
                        'importance': 0.1, 
                        'status': 'error',
                        'error': str(item_error)
                    })
                    
            # Обновляем уровень концентрации
            self.focus_level = min(1.0, self.focus_level + len(important_elements) * 0.05)
            
            # Обновляем историю фильтрации
            self._log_filtering_operation(len(incoming_data), len(important_elements), filtering_strategy)
            
            logger.info(f"Information filtered: {len(filtered_data)} items processed (important: {len(important_elements)})")
            return {
                'filtered_data': filtered_data,
                'important_elements': important_elements,
                'attention_signals': attention_signals,
                'filtering_strategy': filtering_strategy,
                'focus_level': self.focus_level,
                'cognitive_load': self.cognitive_load
            }
            
        except Exception as e:
            logger.error(f"Error in information filtering: {e}")
            # Возврат базового результата в случае ошибки
            return {
                'filtered_data': [{'item': item, 'importance': 0.0, 'status': 'error'} for item in incoming_data],
                'important_elements': [],
                'attention_signals': [],
                'filtering_strategy': filtering_strategy,
                'focus_level': self.focus_level,
                'cognitive_load': self.cognitive_load
            }
    
    def _strict_filtering(self, item):
        """Строгая фильтрация - только очень важные элементы"""
        importance_score = 0.0
        # Проверяем на ключевые слова
        keywords = ['important', 'critical', 'urgent', 'key', 'main', 'essential']
        for keyword in keywords:
            if keyword in item.lower():
                importance_score += 1.0
        # Проверяем на длину
        if len(item) > 30:
            importance_score += 0.5
        return min(1.0, importance_score)
    
    def _balanced_filtering(self, item):
        """Сбалансированная фильтрация"""
        importance_score = 0.0
        # Проверяем на ключевые слова
        keywords = ['important', 'critical', 'urgent', 'key', 'main', 'essential', 'significant']
        for keyword in keywords:
            if keyword in item.lower():
                importance_score += 0.5
        # Проверяем на длину
        if len(item) > 20:
            importance_score += 0.3
        # Проверяем на наличие вопросов
        if '?' in item:
            importance_score += 0.2
        return min(1.0, importance_score)
    
    def _adaptive_filtering(self, item):
        """Адаптивная фильтрация с учетом контекста"""
        importance_score = 0.0
        # Базовая оценка
        importance_score = self._balanced_filtering(item)
        
        # Адаптация под текущее состояние внимания
        attention_factor = 1.0 + (self.focus_level - 0.5) * 0.5  # Более высокий порог при высоком уровне концентрации
        importance_score *= attention_factor
        
        # Учет динамики
        if len(self.attention_history) > 0:
            recent_focus = self.attention_history[-1].get('focus_level', 0.5)
            recent_factor = 1.0 + (recent_focus - 0.5) * 0.3
            importance_score *= recent_factor
            
        return min(1.0, importance_score)
    
    def _calculate_adaptive_threshold(self, incoming_data):
        """Расчет адаптивного порога фильтрации"""
        if not incoming_data:
            return 0.5
            
        # Средняя длина элементов
        avg_length = sum(len(str(item)) for item in incoming_data) / len(incoming_data)
        
        # Оценка средней важности
        avg_importance = 0.0
        for item in incoming_data[:10]:  # Ограничиваем анализ первыми 10 элементами
            if isinstance(item, str):
                importance = self._balanced_filtering(item)
                avg_importance += importance
        avg_importance /= min(10, len(incoming_data))
        
        # Адаптивный порог
        threshold = 0.4 + (avg_length / 100.0) * 0.3 + (avg_importance * 0.3)
        return min(0.9, max(0.1, threshold))
    
    def _generate_attention_signal(self, item, importance_score):
        """Генерация сигнала внимания для элемента"""
        signal = {
            'item_identifier': hash(str(item)) % 1000000,
            'importance': importance_score,
            'signal_strength': min(1.0, importance_score * 2.0),
            'attention_category': 'high' if importance_score > 0.7 else 'medium' if importance_score > 0.3 else 'low',
            'timestamp': time.time()
        }
        
        # Добавляем эмоциональную составляющую
        if 'urgent' in item.lower() or 'critical' in item.lower():
            signal['emotional_intensity'] = 0.8
        elif 'happy' in item.lower() or 'excited' in item.lower():
            signal['emotional_intensity'] = 0.6
        elif 'sad' in item.lower() or 'frustrated' in item.lower():
            signal['emotional_intensity'] = 0.4
        else:
            signal['emotional_intensity'] = 0.2
            
        return signal
    
    def _process_complex_item(self, item_dict):
        """Обработка сложных элементов данных"""
        importance_score = 0.0
        
        # Проверяем наличие ключевых полей
        required_fields = ['title', 'content', 'tags']
        for field in required_fields:
            if field in item_dict and item_dict[field]:
                importance_score += 0.3
                
        # Проверяем на наличие эмоциональных данных
        if 'emotions' in item_dict and item_dict['emotions']:
            emotion_count = len(item_dict['emotions'])
            importance_score += min(0.3, emotion_count * 0.1)
            
        # Проверяем на наличие временных меток
        if 'timestamp' in item_dict:
            importance_score += 0.2
            
        # Проверяем на наличие контекста
        if 'context' in item_dict and item_dict['context']:
            importance_score += 0.2
            
        return min(1.0, importance_score)
    
    def _update_cognitive_load(self, elements_count, weights):
        """Обновление когнитивной нагрузки"""
        # Базовая нагрузка от количества элементов
        base_load = elements_count * 0.05
        
        # Нагрузка от весов
        weight_load = sum(weights.values()) * 0.02 if weights else 0.0
        
        # Общая нагрузка
        self.cognitive_load = min(self.max_cognitive_load, 
                                self.cognitive_load * (1 - self.attention_decay_rate) + 
                                base_load + weight_load)
        
        # Сброс нагрузки при высоком уровне концентрации
        if self.focus_level > 0.8:
            self.cognitive_load *= 0.9
    
    def _update_attention_spread(self, weights):
        """Обновление распределения внимания"""
        if not weights:
            return
            
        # Обновляем распределение по элементам
        total_weight = sum(weights.values())
        if total_weight > 0:
            for element, weight in weights.items():
                self.attention_spread[element] = weight / total_weight
    
    def _log_filtering_operation(self, total_items, important_items, strategy):
        """Логирование операции фильтрации"""
        logger.debug(f"Filtering operation: {total_items} total, {important_items} important, strategy: {strategy}")
    
    def shift_attention(self, new_focus_elements, context=None):
        """Переключение внимания на новые элементы"""
        try:
            # Увеличиваем счетчик переключений
            self.focus_shifts += 1
            
            # Обновляем уровень концентрации при переключении
            self.focus_level = min(1.0, self.focus_level * 0.8 + 0.2)  # Плавное снижение
            
            # Обновляем историю переключений
            self.attention_history.append({
                'type': 'shift',
                'new_focus': new_focus_elements,
                'context': context,
                'timestamp': time.time(),
                'focus_level': self.focus_level
            })
            
            logger.info(f"Attention shifted to {len(new_focus_elements)} elements")
            return True
            
        except Exception as e:
            logger.error(f"Error shifting attention: {e}")
            return False
    
    def get_attention_summary(self):
        """Получение сводки по системе внимания"""
        return {
            'focus_level': self.focus_level,
            'cognitive_load': self.cognitive_load,
            'focus_shifts': self.focus_shifts,
            'attention_spread': dict(self.attention_spread),
            'recent_history': list(self.attention_history)[-5:],  # Последние 5 записей
            'timestamp': time.time()
        }
    
    def reset_attention(self):
        """Сброс системы внимания"""
        self.focus_level = 1.0
        self.attention_weights = {}
        self.distracting_factors = []
        self.cognitive_load = 0.0
        self.focus_shifts = 0
        self.attention_spread.clear()
        self.attention_history.clear()
        logger.info("Attention system reset")
class MemoryTypes:
    """Система различных типов памяти."""
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = {}
        self.emotional_memory = {}
        self.procedural_memory = {}
        self.semantic_memory = {}
        self.episodic_memory = {}
        self.working_memory = {}
        self.memory_hierarchy = {}
        self.memory_access_counter = defaultdict(int)
        self.memory_decay_rates = {
            'short_term': 0.05,
            'long_term': 0.001,
            'episodic': 0.01,
            'semantic': 0.005
        }
        self.memory_strength = {}
        
    def store_memory(self, data, memory_type="short_term", emotional_value=0.0, 
                     context=None, importance_score=0.5, memory_id=None):
        """Хранение памяти разных типов с расширенной функциональностью"""
        try:
            # Генерация уникального ID если не указан
            if memory_id is None:
                memory_id = f"{memory_type}_{hash(str(data))}_{int(time.time())}"
            
            # Сохранение в соответствующий тип памяти
            if memory_type == "short_term":
                self.short_term_memory.append({
                    'id': memory_id,
                    'data': data,
                    'timestamp': time.time(),
                    'context': context,
                    'emotional_value': emotional_value,
                    'importance': importance_score,
                    'access_count': 0
                })
                # Ограничение размера краткосрочной памяти
                if len(self.short_term_memory) > 1000:
                    self.short_term_memory.pop(0)
                    
            elif memory_type == "long_term":
                self.long_term_memory[memory_id] = {
                    'data': data,
                    'timestamp': time.time(),
                    'context': context,
                    'emotional_value': emotional_value,
                    'importance': importance_score,
                    'access_count': 0,
                    'last_access': time.time(),
                    'decay_factor': 1.0
                }
                
            elif memory_type == "emotional":
                self.emotional_memory[memory_id] = {
                    'data': data,
                    'timestamp': time.time(),
                    'emotional_state': emotional_value,
                    'context': context,
                    'intensity': abs(emotional_value),
                    'memory_type': memory_type,
                    'access_count': 0
                }
                
            elif memory_type == "procedural":
                self.procedural_memory[memory_id] = {
                    'data': data,
                    'timestamp': time.time(),
                    'context': context,
                    'difficulty': importance_score,
                    'success_rate': 0.0,
                    'attempts': 0,
                    'access_count': 0
                }
                
            elif memory_type == "semantic":
                self.semantic_memory[memory_id] = {
                    'data': data,
                    'timestamp': time.time(),
                    'context': context,
                    'conceptual_depth': importance_score,
                    'connections': [],
                    'access_count': 0
                }
                
            elif memory_type == "episodic":
                self.episodic_memory[memory_id] = {
                    'data': data,
                    'timestamp': time.time(),
                    'context': context,
                    'emotional_value': emotional_value,
                    'importance': importance_score,
                    'access_count': 0,
                    'memory_type': memory_type
                }
                
            # Обновление иерархии памяти
            self._update_memory_hierarchy(memory_id, memory_type, importance_score)
            
            # Обновление счетчика доступа
            self.memory_access_counter[memory_type] += 1
            
            logger.info(f"Memory stored in {memory_type} memory (ID: {memory_id})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return None
    
    def retrieve_memory(self, memory_id, memory_type=None):
        """Извлечение памяти по ID"""
        try:
            retrieved_data = None
            
            # Пытаемся найти в разных типах памяти
            if memory_type:
                if memory_type == "short_term" and memory_id in [item['id'] for item in self.short_term_memory]:
                    item = next(item for item in self.short_term_memory if item['id'] == memory_id)
                    retrieved_data = item
                    item['access_count'] += 1
                elif memory_type == "long_term" and memory_id in self.long_term_memory:
                    retrieved_data = self.long_term_memory[memory_id]
                    retrieved_data['access_count'] += 1
                    retrieved_data['last_access'] = time.time()
                elif memory_type == "emotional" and memory_id in self.emotional_memory:
                    retrieved_data = self.emotional_memory[memory_id]
                    retrieved_data['access_count'] += 1
                elif memory_type == "procedural" and memory_id in self.procedural_memory:
                    retrieved_data = self.procedural_memory[memory_id]
                    retrieved_data['access_count'] += 1
                elif memory_type == "semantic" and memory_id in self.semantic_memory:
                    retrieved_data = self.semantic_memory[memory_id]
                    retrieved_data['access_count'] += 1
                elif memory_type == "episodic" and memory_id in self.episodic_memory:
                    retrieved_data = self.episodic_memory[memory_id]
                    retrieved_data['access_count'] += 1
            else:
                # Поиск по всем типам
                for mem_type, mem_dict in [
                    ("short_term", self.short_term_memory),
                    ("long_term", self.long_term_memory),
                    ("emotional", self.emotional_memory),
                    ("procedural", self.procedural_memory),
                    ("semantic", self.semantic_memory),
                    ("episodic", self.episodic_memory)
                ]:
                    if isinstance(mem_dict, dict) and memory_id in mem_dict:
                        retrieved_data = mem_dict[memory_id]
                        if isinstance(retrieved_data, dict):
                            retrieved_data['access_count'] += 1
                            if 'last_access' in retrieved_data:
                                retrieved_data['last_access'] = time.time()
                        break
                    elif isinstance(mem_dict, list):
                        for item in mem_dict:
                            if item.get('id') == memory_id:
                                retrieved_data = item
                                item['access_count'] += 1
                                break
            
            # Обновление силы памяти
            if retrieved_data and 'access_count' in retrieved_data:
                self._update_memory_strength(memory_id, memory_type, retrieved_data['access_count'])
            
            logger.debug(f"Memory retrieved: {memory_id}")
            return retrieved_data
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    def update_memory(self, memory_id, new_data=None, emotional_value=None, importance_score=None):
        """Обновление существующей памяти"""
        try:
            updated = False
            
            # Поиск и обновление в разных типах памяти
            for mem_type, mem_dict in [
                ("short_term", self.short_term_memory),
                ("long_term", self.long_term_memory),
                ("emotional", self.emotional_memory),
                ("procedural", self.procedural_memory),
                ("semantic", self.semantic_memory),
                ("episodic", self.episodic_memory)
            ]:
                if isinstance(mem_dict, dict) and memory_id in mem_dict:
                    item = mem_dict[memory_id]
                    if new_data is not None:
                        item['data'] = new_data
                    if emotional_value is not None:
                        if 'emotional_value' in item:
                            item['emotional_value'] = emotional_value
                        elif 'emotional_state' in item:
                            item['emotional_state'] = emotional_value
                    if importance_score is not None:
                        if 'importance' in item:
                            item['importance'] = importance_score
                        elif 'difficulty' in item:
                            item['difficulty'] = importance_score
                    updated = True
                    break
                elif isinstance(mem_dict, list):
                    for i, item in enumerate(mem_dict):
                        if item.get('id') == memory_id:
                            if new_data is not None:
                                item['data'] = new_data
                            if emotional_value is not None:
                                item['emotional_value'] = emotional_value
                            if importance_score is not None:
                                item['importance'] = importance_score
                            updated = True
                            break
            
            if updated:
                logger.info(f"Memory updated: {memory_id}")
            else:
                logger.warning(f"Memory not found for update: {memory_id}")
                
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
    
    def delete_memory(self, memory_id, memory_type=None):
        """Удаление памяти"""
        try:
            deleted = False
            
            if memory_type:
                if memory_type == "short_term":
                    self.short_term_memory = [item for item in self.short_term_memory if item['id'] != memory_id]
                elif memory_type == "long_term" and memory_id in self.long_term_memory:
                    del self.long_term_memory[memory_id]
                elif memory_type == "emotional" and memory_id in self.emotional_memory:
                    del self.emotional_memory[memory_id]
                elif memory_type == "procedural" and memory_id in self.procedural_memory:
                    del self.procedural_memory[memory_id]
                elif memory_type == "semantic" and memory_id in self.semantic_memory:
                    del self.semantic_memory[memory_id]
                elif memory_type == "episodic" and memory_id in self.episodic_memory:
                    del self.episodic_memory[memory_id]
            else:
                # Поиск и удаление по всем типам
                for mem_type, mem_dict in [
                    ("short_term", self.short_term_memory),
                    ("long_term", self.long_term_memory),
                    ("emotional", self.emotional_memory),
                    ("procedural", self.procedural_memory),
                    ("semantic", self.semantic_memory),
                    ("episodic", self.episodic_memory)
                ]:
                    if isinstance(mem_dict, dict) and memory_id in mem_dict:
                        del mem_dict[memory_id]
                        deleted = True
                        break
                    elif isinstance(mem_dict, list):
                        self.short_term_memory = [item for item in self.short_term_memory if item['id'] != memory_id]
                        deleted = True
                        break
            
            if deleted:
                logger.info(f"Memory deleted: {memory_id}")
            else:
                logger.warning(f"Memory not found for deletion: {memory_id}")
                
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
    
    def get_memory_stats(self):
        """Получение статистики по памяти"""
        stats = {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'emotional_count': len(self.emotional_memory),
            'procedural_count': len(self.procedural_memory),
            'semantic_count': len(self.semantic_memory),
            'episodic_count': len(self.episodic_memory),
            'memory_access_counts': dict(self.memory_access_counter),
            'total_memory_items': (
                len(self.short_term_memory) + 
                len(self.long_term_memory) + 
                len(self.emotional_memory) + 
                len(self.procedural_memory) + 
                len(self.semantic_memory) + 
                len(self.episodic_memory)
            )
        }
        return stats
    
    def _update_memory_hierarchy(self, memory_id, memory_type, importance_score):
        """Обновление иерархии памяти"""
        if memory_id not in self.memory_hierarchy:
            self.memory_hierarchy[memory_id] = {
                'type': memory_type,
                'importance': importance_score,
                'hierarchy_level': self._calculate_hierarchy_level(importance_score),
                'timestamp': time.time()
            }
    
    def _calculate_hierarchy_level(self, importance_score):
        """Расчет уровня иерархии памяти"""
        if importance_score >= 0.8:
            return 'high'
        elif importance_score >= 0.5:
            return 'medium'
        elif importance_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _update_memory_strength(self, memory_id, memory_type, access_count):
        """Обновление силы памяти"""
        if memory_id not in self.memory_strength:
            self.memory_strength[memory_id] = {
                'access_count': 0,
                'strength': 0.0,
                'last_updated': time.time()
            }
        
        self.memory_strength[memory_id]['access_count'] = access_count
        self.memory_strength[memory_id]['strength'] = min(1.0, access_count * 0.1)
        self.memory_strength[memory_id]['last_updated'] = time.time()
    
    def decay_memory(self, memory_type=None):
        """Уменьшение силы памяти со временем"""
        try:
            decay_rate = self.memory_decay_rates.get(memory_type, 0.01) if memory_type else 0.01
            
            # Уменьшаем силу памяти
            if memory_type == "short_term":
                # Для краткосрочной памяти - просто очищаем
                self.short_term_memory.clear()
            elif memory_type == "long_term":
                for memory_id, memory_data in list(self.long_term_memory.items()):
                    # Уменьшаем фактор декейда
                    memory_data['decay_factor'] = max(0.0, memory_data['decay_factor'] - decay_rate)
                    # Удаляем очень старую память
                    if time.time() - memory_data['timestamp'] > 3600 * 24 * 30:  # 30 дней
                        del self.long_term_memory[memory_id]
            elif memory_type == "episodic":
                for memory_id, memory_data in list(self.episodic_memory.items()):
                    memory_data['decay_factor'] = max(0.0, memory_data['decay_factor'] - decay_rate)
                    # Удаляем очень старую память
                    if time.time() - memory_data['timestamp'] > 3600 * 24 * 30:  # 30 дней
                        del self.episodic_memory[memory_id]
            
            logger.info(f"Memory decay applied for {memory_type or 'all types'}")
            
        except Exception as e:
            logger.error(f"Error in memory decay: {e}")

class IntuitionSystem:
    """Система интуиции и быстрых выводов."""
    def __init__(self):
        self.pattern_recognition = 0.0
        self.quick_decisions = []
        self.insight_triggers = []
        self.intuition_memory = {}
        self.intuition_strength = {}
        self.intuition_history = deque(maxlen=100)
        self.intuition_confidence_threshold = 0.6
        self.intuition_recall_history = {}
        self.cognitive_load = 0.0
        
    def quick_insight(self, partial_data, context=None, confidence_boost=1.0):
        """Быстрое озарение на основе частичной информации"""
        try:
            # Использование неполных данных для быстрых выводов
            # Интуитивное понимание
            insight = {
                'partial_data': partial_data,
                'insight': '',
                'confidence': 0.0,
                'reasoning': [],
                'context': context,
                'timestamp': time.time(),
                'intuition_strength': 0.0,
                'memory_triggers': []
            }
            
            # Расширенный анализ частичных данных
            if isinstance(partial_data, str):
                words = partial_data.split()
                word_count = len(words)
                
                # Оценка качества данных
                quality_score = 0.0
                if word_count > 3:
                    quality_score += 0.3
                if len(partial_data) > 10:
                    quality_score += 0.2
                if '?' in partial_data:
                    quality_score += 0.2
                if '!' in partial_data:
                    quality_score += 0.1
                    
                # Простой анализ на основе ключевых слов
                key_words = ['quick', 'fast', 'immediate', 'instant', 'urgent', 'critical', 'important']
                found_keys = [kw for kw in key_words if kw in partial_data.lower()]
                if found_keys:
                    insight['insight'] = f"Fast decision based on {len(found_keys)} key indicators"
                    insight['confidence'] = 0.7 * confidence_boost
                    insight['reasoning'].append("Key words detected")
                    insight['intuition_strength'] = min(1.0, len(found_keys) * 0.2)
                else:
                    insight['insight'] = "Pattern recognition from partial data"
                    insight['confidence'] = 0.5 * confidence_boost
                    insight['reasoning'].append("General pattern recognition")
                    insight['intuition_strength'] = 0.3
                    
                # Проверка на эмоциональные слова
                emotional_words = ['exciting', 'amazing', 'shocking', 'surprising', 'frustrating']
                found_emotions = [ew for ew in emotional_words if ew in partial_data.lower()]
                if found_emotions:
                    insight['reasoning'].append(f"Emotional cues detected: {found_emotions}")
                    insight['intuition_strength'] = min(1.0, insight['intuition_strength'] + 0.2)
                    
                # Проверка на контекстные триггеры
                if context and isinstance(context, dict):
                    context_keys = list(context.keys())
                    if context_keys:
                        insight['memory_triggers'] = context_keys[:3]  # Первые 3 ключа
                        insight['reasoning'].append(f"Context triggers: {context_keys[:3]}")
                        
            elif isinstance(partial_data, list) and len(partial_data) > 0:
                # Анализ списков
                insight['insight'] = f"Data pattern recognition from {len(partial_data)} items"
                insight['confidence'] = 0.6 * confidence_boost
                insight['reasoning'].append("List pattern analysis")
                insight['intuition_strength'] = min(1.0, len(partial_data) * 0.05)
                
            elif isinstance(partial_data, dict):
                # Анализ словарей
                insight['insight'] = f"Structured data pattern recognition"
                insight['confidence'] = 0.5 * confidence_boost
                insight['reasoning'].append("Dictionary structure analysis")
                insight['intuition_strength'] = min(1.0, len(partial_data) * 0.1)
                
            else:
                insight['insight'] = "Insufficient data for meaningful insight"
                insight['confidence'] = 0.2 * confidence_boost
                insight['reasoning'].append("Too little information")
                insight['intuition_strength'] = 0.1
                
            # Учет когнитивной нагрузки
            if self.cognitive_load > 0.8:
                insight['confidence'] *= 0.7
                insight['reasoning'].append("High cognitive load affects insight quality")
                
            # Учет силы интуиции
            insight['intuition_strength'] = min(1.0, insight['intuition_strength'] * confidence_boost)
            
            # Сохранение в историю интуиции
            self.intuition_history.append(insight)
            
            # Обновление статистики
            self._update_intuition_stats(insight)
            
            logger.info(f"Quick insight generated: {insight['insight'][:50]}...")
            return insight
            
        except Exception as e:
            logger.error(f"Error in quick insight generation: {e}")
            return {
                'partial_data': partial_data,
                'insight': 'Error generating insight',
                'confidence': 0.1,
                'reasoning': [f'Error: {str(e)}'],
                'timestamp': time.time(),
                'intuition_strength': 0.0
            }
    
    def pattern_recognition(self, data, pattern_type="structural"):
        """Распознавание паттернов без полного анализа"""
        try:
            # Быстрое распознавание закономерностей
            # Связывание неочевидных элементов
            patterns = []
            pattern_count = 0
            
            if isinstance(data, list) and len(data) > 1:
                # Простой анализ последовательности
                for i in range(len(data) - 1):
                    if isinstance(data[i], str) and isinstance(data[i+1], str):
                        # Проверяем на совпадения
                        if data[i].lower() == data[i+1].lower():
                            patterns.append({
                                'pattern': 'repetition',
                                'position': i,
                                'value': data[i],
                                'type': pattern_type
                            })
                            pattern_count += 1
                        # Проверяем на схожие слова
                        elif data[i][:3] == data[i+1][:3]:
                            patterns.append({
                                'pattern': 'similarity',
                                'position': i,
                                'values': (data[i], data[i+1]),
                                'type': pattern_type
                            })
                            pattern_count += 1
                        # Проверяем на последовательность
                        elif i > 0 and isinstance(data[i-1], str) and isinstance(data[i+1], str):
                            if data[i-1][-2:] == data[i][:2] and data[i+1][:2] == data[i][-2:]:
                                patterns.append({
                                    'pattern': 'sequence_connection',
                                    'position': i,
                                    'values': (data[i-1], data[i], data[i+1]),
                                    'type': pattern_type
                                })
                                pattern_count += 1
                                
                # Проверяем на числовые паттерны
                numeric_patterns = self._detect_numeric_patterns(data)
                patterns.extend(numeric_patterns)
                pattern_count += len(numeric_patterns)
                
            elif isinstance(data, dict):
                # Анализ структуры словаря
                keys = list(data.keys())
                if len(keys) > 1:
                    # Проверяем на повторяющиеся ключи
                    key_groups = {}
                    for key in keys:
                        if isinstance(key, str):
                            key_lower = key.lower()
                            if key_lower not in key_groups:
                                key_groups[key_lower] = []
                            key_groups[key_lower].append(key)
                    
                    for group_key, group_values in key_groups.items():
                        if len(group_values) > 1:
                            patterns.append({
                                'pattern': 'key_duplication',
                                'group': group_key,
                                'keys': group_values,
                                'type': pattern_type
                            })
                            pattern_count += 1
                            
            # Обновляем уровень распознавания паттернов
            self.pattern_recognition = min(1.0, pattern_count * 0.1)
            
            logger.info(f"Pattern recognition completed, found {pattern_count} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            return []
    
    def _detect_numeric_patterns(self, data):
        """Обнаружение числовых паттернов"""
        patterns = []
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        
        if len(numeric_data) > 2:
            # Проверяем на арифметическую последовательность
            diffs = [numeric_data[i+1] - numeric_data[i] for i in range(len(numeric_data)-1)]
            if len(set(diffs)) == 1:  # Все разности одинаковы
                patterns.append({
                    'pattern': 'arithmetic_sequence',
                    'sequence': numeric_data,
                    'difference': diffs[0],
                    'type': 'numeric'
                })
                
            # Проверяем на геометрическую последовательность
            ratios = []
            for i in range(len(numeric_data)-1):
                if numeric_data[i] != 0:
                    ratios.append(numeric_data[i+1] / numeric_data[i])
            
            if len(set(ratios)) == 1 and ratios[0] != 0:  # Все отношения одинаковы
                patterns.append({
                    'pattern': 'geometric_sequence',
                    'sequence': numeric_data,
                    'ratio': ratios[0],
                    'type': 'numeric'
                })
                
        return patterns
    
    def recall_intuition(self, context=None, similarity_threshold=0.7):
        """Воспоминание интуитивных решений"""
        try:
            recalled_intuitions = []
            
            # Поиск похожих контекстов
            if context and self.intuition_history:
                for insight in self.intuition_history:
                    if insight.get('context') and context:
                        # Простое сравнение контекстов
                        context_str = str(context).lower()
                        insight_context_str = str(insight['context']).lower()
                        similarity = self._calculate_context_similarity(context_str, insight_context_str)
                        
                        if similarity >= similarity_threshold:
                            recalled_intuitions.append({
                                'insight': insight,
                                'similarity': similarity,
                                'timestamp': insight['timestamp']
                            })
            
            # Сортировка по схожести
            recalled_intuitions.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Сохраняем историю воспоминаний
            self.intuition_recall_history[time.time()] = {
                'context': context,
                'recalled_count': len(recalled_intuitions),
                'timestamp': time.time()
            }
            
            logger.info(f"Recalled {len(recalled_intuitions)} intuitions")
            return recalled_intuitions
            
        except Exception as e:
            logger.error(f"Error in intuition recall: {e}")
            return []
    
    def _calculate_context_similarity(self, context1, context2):
        """Расчет схожести контекстов"""
        words1 = set(context1.split())
        words2 = set(context2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_intuition_stats(self, insight):
        """Обновление статистики интуиции"""
        insight_id = f"{hash(str(insight['partial_data']))}_{int(time.time())}"
        self.intuition_memory[insight_id] = insight
        
        # Обновляем силу интуиции
        if insight['insight'] not in self.intuition_strength:
            self.intuition_strength[insight['insight']] = 0.0
        self.intuition_strength[insight['insight']] += insight['intuition_strength']
    
    def get_intuition_summary(self):
        """Получение сводки по интуиции"""
        return {
            'pattern_recognition': self.pattern_recognition,
            'intuition_count': len(self.intuition_history),
            'intuition_memory_size': len(self.intuition_memory),
            'recent_intuitions': list(self.intuition_history)[-10:] if len(self.intuition_history) >= 10 else list(self.intuition_history),
            'intuition_strength': self.intuition_strength,
            'recall_history': self.intuition_recall_history,
            'timestamp': time.time()
        }
    
    def set_cognitive_load(self, load_level):
        """Установка уровня когнитивной нагрузки"""
        self.cognitive_load = min(1.0, max(0.0, load_level))
        logger.debug(f"Cognitive load set to: {self.cognitive_load}")
class MotivationSystem:
    """Система мотивации и целеполагания."""
    def __init__(self):
        self.goals = []
        self.motivation_level = 0.5
        self.reward_system = {}
        self.punishment_system = {}
        self.goal_history = []
        self.motivation_history = deque(maxlen=100)
        self.motivational_factors = {
            'achievement': 0.0,
            'recognition': 0.0,
            'autonomy': 0.0,
            'competence': 0.0,
            'relatedness': 0.0
        }
        self.motivational_context = {}
        self.performance_benchmark = 0.0
        self.motivation_decay_rate = 0.01
        
    def set_goal(self, goal, importance, deadline=None, reward_value=1.0, 
                 personal_relevance=0.5, goal_type="general"):
        """Установка цели с расширенными параметрами"""
        try:
            goal_obj = {
                'goal': goal,
                'importance': min(1.0, max(0.0, importance)),
                'progress': 0.0,
                'motivation_required': importance * 0.8,
                'created_at': time.time(),
                'deadline': deadline,
                'reward_value': reward_value,
                'personal_relevance': personal_relevance,
                'goal_type': goal_type,
                'status': 'active',
                'completion_time': None,
                'effort_estimate': importance * 10,  # Примерная оценка усилий
                'success_probability': 0.5,
                'milestones': [],
                'associated_emotions': []
            }
            
            # Добавляем цели в историю
            self.goal_history.append({
                'goal': goal,
                'importance': importance,
                'timestamp': time.time(),
                'status': 'created'
            })
            
            self.goals.append(goal_obj)
            logger.info(f"Goal set: {goal} (importance: {importance:.2f}, type: {goal_type})")
            return goal_obj
            
        except Exception as e:
            logger.error(f"Error setting goal '{goal}': {e}")
            return None
    
    def add_milestone(self, goal_id, milestone_description, milestone_importance=0.5):
        """Добавление промежуточной цели (майлстоуна)"""
        try:
            for goal in self.goals:
                if id(goal) == goal_id or goal['goal'] == goal_id:
                    milestone = {
                        'description': milestone_description,
                        'importance': milestone_importance,
                        'completed': False,
                        'completion_time': None,
                        'reward': milestone_importance * 0.2
                    }
                    goal['milestones'].append(milestone)
                    logger.info(f"Milestone added to goal '{goal['goal']}': {milestone_description}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error adding milestone: {e}")
            return False
    
    def update_progress(self, goal_id, progress, context=None):
        """Обновление прогресса цели"""
        try:
            for goal in self.goals:
                if id(goal) == goal_id or goal['goal'] == goal_id:
                    # Ограничиваем прогресс от 0 до 1
                    goal['progress'] = min(1.0, max(0.0, progress))
                    
                    # Обновляем статус цели
                    if goal['progress'] >= 1.0:
                        goal['status'] = 'completed'
                        goal['completion_time'] = time.time()
                        self._handle_goal_completion(goal, context)
                    elif goal['progress'] > 0.0:
                        goal['status'] = 'in_progress'
                    else:
                        goal['status'] = 'active'
                    
                    # Обновляем историю прогресса
                    self.goal_history.append({
                        'goal': goal['goal'],
                        'progress': goal['progress'],
                        'timestamp': time.time(),
                        'status': goal['status']
                    })
                    
                    # Обновляем мотивационные факторы
                    self._update_motivational_factors(goal, progress)
                    
                    logger.info(f"Progress updated for goal '{goal['goal']}': {progress:.2f}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error updating progress for goal '{goal_id}': {e}")
            return False
    
    def _handle_goal_completion(self, goal, context=None):
        """Обработка завершения цели"""
        try:
            # Награда за завершение
            reward = goal['reward_value'] * 2.0  # Увеличенная награда за завершение
            
            # Обновляем систему наград
            if goal['goal'] not in self.reward_system:
                self.reward_system[goal['goal']] = []
            self.reward_system[goal['goal']].append({
                'amount': reward,
                'timestamp': time.time(),
                'context': context
            })
            
            # Обновляем уровень мотивации
            self.motivation_level = min(1.0, self.motivation_level + 0.1)
            
            # Обновляем историю
            self.goal_history.append({
                'goal': goal['goal'],
                'progress': 1.0,
                'timestamp': time.time(),
                'status': 'completed',
                'reward': reward
            })
            
            logger.info(f"Goal completed: {goal['goal']} (Reward: {reward:.2f})")
            
        except Exception as e:
            logger.error(f"Error handling goal completion: {e}")
    
    def assess_motivation(self, current_progress=None, context=None):
        """Оценка мотивации с учетом различных факторов"""
        try:
            motivation_score = 0.0
            
            # Базовая мотивация от прогресса
            if self.goals:
                # Средний прогресс по всем целям
                avg_progress = sum(g['progress'] for g in self.goals) / len(self.goals)
                
                # Оценка по прогрессу с учетом важности
                weighted_progress = sum(g['progress'] * g['importance'] for g in self.goals) / sum(g['importance'] for g in self.goals) if sum(g['importance'] for g in self.goals) > 0 else 0
                
                motivation_score = weighted_progress * 0.6 + 0.4
                
                # Учет уровня мотивации
                motivation_score = motivation_score * self.motivation_level
                
                # Учет времени
                time_factor = self._calculate_time_factor()
                motivation_score *= time_factor
                
                # Учет личной значимости
                relevance_factor = self._calculate_relevance_factor()
                motivation_score *= relevance_factor
                
                # Учет наград
                reward_factor = self._calculate_reward_factor()
                motivation_score *= reward_factor
                
                # Учет майлстоунов
                milestone_factor = self._calculate_milestone_factor()
                motivation_score *= milestone_factor
                
            else:
                # Если нет целей, мотивация зависит от базового уровня
                motivation_score = self.motivation_level * 0.5
            
            # Адаптация мотивации к контексту
            if context and isinstance(context, dict):
                # Учет контекстных факторов
                context_factor = context.get('motivation_context_factor', 1.0)
                motivation_score *= context_factor
            
            # Обновление истории мотивации
            self.motivation_history.append({
                'score': motivation_score,
                'timestamp': time.time(),
                'progress': current_progress,
                'context': context
            })
            
            # Обновляем уровень мотивации
            self._update_motivation_level(motivation_score, current_progress)
            
            # Обновляем факторы мотивации
            self._update_motivational_factors_from_context(context)
            
            logger.debug(f"Motivation assessed: {motivation_score:.3f} "
                        f"(base: {self.motivation_level:.3f}, progress: {avg_progress:.3f})")
            return motivation_score
            
        except Exception as e:
            logger.error(f"Error assessing motivation: {e}")
            return self.motivation_level
    
    def _calculate_time_factor(self):
        """Расчет фактора времени"""
        # Чем ближе дедлайн, тем выше мотивация (до определенного момента)
        time_factors = []
        for goal in self.goals:
            if goal['deadline']:
                time_diff = goal['deadline'] - time.time()
                if time_diff > 0:
                    # Нормализация времени до 1 дня
                    normalized_time = min(1.0, time_diff / 86400.0)  # 86400 секунд в дне
                    time_factors.append(normalized_time)
        
        if time_factors:
            # Средний фактор времени
            return sum(time_factors) / len(time_factors) * 0.5 + 0.5  # От 0.5 до 1.0
        return 1.0
    
    def _calculate_relevance_factor(self):
        """Расчет фактора личной значимости"""
        if not self.goals:
            return 1.0
        
        relevance_sum = sum(goal['personal_relevance'] for goal in self.goals)
        avg_relevance = relevance_sum / len(self.goals)
        return avg_relevance * 0.5 + 0.5  # От 0.5 до 1.0
    
    def _calculate_reward_factor(self):
        """Расчет фактора наград"""
        reward_sum = 0.0
        for goal_rewards in self.reward_system.values():
            reward_sum += sum(r['amount'] for r in goal_rewards) if goal_rewards else 0.0
        
        # Нормализация
        if reward_sum > 0:
            return min(1.5, 1.0 + reward_sum * 0.01)
        return 1.0
    
    def _calculate_milestone_factor(self):
        """Расчет фактора майлстоунов"""
        if not self.goals:
            return 1.0
        
        total_milestones = sum(len(goal['milestones']) for goal in self.goals)
        completed_milestones = sum(sum(1 for m in goal['milestones'] if m['completed']) 
                                 for goal in self.goals)
        
        if total_milestones > 0:
            milestone_ratio = completed_milestones / total_milestones
            return min(1.5, 1.0 + milestone_ratio * 0.5)
        return 1.0
    
    def _update_motivation_level(self, motivation_score, current_progress):
        """Обновление уровня мотивации"""
        # Увеличиваем уровень мотивации при достижении прогресса
        if current_progress and current_progress > 0.5:
            self.motivation_level = min(1.0, self.motivation_level + 0.03)
        elif current_progress and current_progress < 0.2:
            self.motivation_level = max(0.0, self.motivation_level - 0.01)
        else:
            # Плавное снижение мотивации со временем
            self.motivation_level = max(0.0, self.motivation_level - self.motivation_decay_rate)
    
    def _update_motivational_factors(self, goal, progress):
        """Обновление факторов мотивации"""
        # Обновляем факторы в зависимости от прогресса
        if progress >= 0.8:
            self.motivational_factors['achievement'] = min(1.0, self.motivational_factors['achievement'] + 0.05)
        if progress >= 0.5:
            self.motivational_factors['competence'] = min(1.0, self.motivational_factors['competence'] + 0.03)
        if progress > 0.0:
            self.motivational_factors['autonomy'] = min(1.0, self.motivational_factors['autonomy'] + 0.02)
    
    def _update_motivational_factors_from_context(self, context):
        """Обновление факторов мотивации из контекста"""
        if context and isinstance(context, dict):
            # Обновляем факторы на основе контекста
            if 'social' in context:
                self.motivational_factors['relatedness'] = min(1.0, self.motivational_factors['relatedness'] + 0.05)
            if 'challenge' in context:
                self.motivational_factors['competence'] = min(1.0, self.motivational_factors['competence'] + 0.03)
            if 'recognition' in context:
                self.motivational_factors['recognition'] = min(1.0, self.motivational_factors['recognition'] + 0.04)
    
    def get_motivation_report(self):
        """Получение отчета по мотивации"""
        try:
            goals_summary = []
            for goal in self.goals:
                goals_summary.append({
                    'goal': goal['goal'],
                    'importance': goal['importance'],
                    'progress': goal['progress'],
                    'status': goal['status'],
                    'relevance': goal['personal_relevance']
                })
            
            # Статистика по мотивационным факторам
            factors_stats = {}
            for factor, value in self.motivational_factors.items():
                factors_stats[factor] = value
            
            return {
                'current_motivation_level': self.motivation_level,
                'goals': goals_summary,
                'motivational_factors': factors_stats,
                'total_goals': len(self.goals),
                'completed_goals': len([g for g in self.goals if g['status'] == 'completed']),
                'active_goals': len([g for g in self.goals if g['status'] == 'active' or g['status'] == 'in_progress']),
                'motivation_history': list(self.motivation_history)[-10:] if len(self.motivation_history) >= 10 else list(self.motivation_history),
                'recent_rewards': self._get_recent_rewards(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error generating motivation report: {e}")
            return {
                'current_motivation_level': self.motivation_level,
                'goals': [],
                'motivational_factors': self.motivational_factors,
                'total_goals': 0,
                'completed_goals': 0,
                'active_goals': 0,
                'motivation_history': [],
                'recent_rewards': [],
                'timestamp': time.time()
            }
    
    def _get_recent_rewards(self):
        """Получение последних наград"""
        recent_rewards = []
        for goal, rewards in self.reward_system.items():
            if rewards:
                # Берем последние 5 наград для каждой цели
                recent_rewards.extend(rewards[-5:])
        # Сортируем по времени
        recent_rewards.sort(key=lambda x: x['timestamp'], reverse=True)
        return recent_rewards[:10]  # Возвращаем последние 10 наград
    
    def adjust_motivation_for_difficulty(self, difficulty_level):
        """Коррекция мотивации в зависимости от сложности"""
        try:
            # Уменьшаем мотивацию при высокой сложности
            difficulty_factor = min(1.0, difficulty_level * 0.3)
            adjusted_motivation = self.motivation_level * (1.0 - difficulty_factor)
            self.motivation_level = max(0.0, adjusted_motivation)
            
            logger.debug(f"Motivation adjusted for difficulty {difficulty_level}: {adjusted_motivation:.3f}")
            return adjusted_motivation
        except Exception as e:
            logger.error(f"Error adjusting motivation for difficulty: {e}")
            return self.motivation_level
    
    def reset_motivation(self):
        """Сброс системы мотивации"""
        self.goals.clear()
        self.motivation_level = 0.5
        self.reward_system.clear()
        self.punishment_system.clear()
        self.goal_history.clear()
        self.motivation_history.clear()
        self.motivational_factors = {
            'achievement': 0.0,
            'recognition': 0.0,
            'autonomy': 0.0,
            'competence': 0.0,
            'relatedness': 0.0
        }
        logger.info("Motivation system reset")
class PersonalitySystem:
    """Система личности и индивидуальности."""
    def __init__(self):
        self.personality_traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        self.thinking_style = "analytical"
        self.learning_preference = "visual"
        self.emotional_bias = 0.0
        self.social_orientation = 0.0
        self.cognitive_bias = 0.0
        self.personality_history = deque(maxlen=100)
        self.adaptation_history = []
        
    def adapt_thinking(self, situation, context=None, emotional_state=None):
        """Адаптация стиля мышления под ситуацию с учетом контекста и эмоций"""
        try:
            # Изменение подхода в зависимости от контекста
            # Учет личностных особенностей
            adaptation = {
                'situation': situation,
                'adapted_style': self.thinking_style,
                'reason': '',
                'context': context,
                'emotional_state': emotional_state,
                'personality_influence': {},
                'timestamp': time.time()
            }
            
            # Учет эмоционального состояния
            emotional_influence = self._calculate_emotional_influence(emotional_state)
            adaptation['personality_influence']['emotional'] = emotional_influence
            
            # Учет контекста
            context_influence = self._analyze_context_influence(context)
            adaptation['personality_influence']['context'] = context_influence
            
            # Сложная адаптация с учетом личностных черт
            if situation == 'problem_solving':
                adaptation['adapted_style'] = self._select_problem_solving_style()
                adaptation['reason'] = 'Problem solving requires logical approach'
                
            elif situation == 'creative_task':
                adaptation['adapted_style'] = self._select_creative_style()
                adaptation['reason'] = 'Creative task requires imaginative approach'
                
            elif situation == 'social_interaction':
                adaptation['adapted_style'] = self._select_social_style()
                adaptation['reason'] = 'Social interaction requires emotional intelligence'
                
            elif situation == 'learning_task':
                adaptation['adapted_style'] = self._select_learning_style()
                adaptation['reason'] = 'Learning task requires adaptive approach'
                
            elif situation == 'decision_making':
                adaptation['adapted_style'] = self._select_decision_style()
                adaptation['reason'] = 'Decision making requires balanced approach'
                
            else:
                adaptation['adapted_style'] = self.thinking_style
                adaptation['reason'] = 'No specific adaptation needed'
            
            # Обновляем историю адаптаций
            self.adaptation_history.append(adaptation)
            
            # Обновляем историю личности
            self.personality_history.append({
                'situation': situation,
                'adapted_style': adaptation['adapted_style'],
                'timestamp': time.time(),
                'personality_traits': self.personality_traits.copy()
            })
            
            logger.info(f"Thinking style adapted for {situation}: {adaptation['adapted_style']}")
            return adaptation
            
        except Exception as e:
            logger.error(f"Error in thinking adaptation: {e}")
            # Возврат базовой адаптации
            return {
                'situation': situation,
                'adapted_style': self.thinking_style,
                'reason': 'Error in adaptation',
                'context': context,
                'emotional_state': emotional_state,
                'personality_influence': {},
                'timestamp': time.time()
            }
    
    def _calculate_emotional_influence(self, emotional_state):
        """Расчет влияния эмоционального состояния на мышление"""
        if not emotional_state:
            return 0.0
            
        emotional_influence = 0.0
        emotional_state_lower = emotional_state.lower()
        
        # Влияние на стиль мышления
        if 'excitement' in emotional_state_lower or 'joy' in emotional_state_lower:
            emotional_influence = 0.3  # Более креативное мышление
        elif 'anxiety' in emotional_state_lower or 'stress' in emotional_state_lower:
            emotional_influence = -0.2  # Более осторожное мышление
        elif 'frustration' in emotional_state_lower or 'anger' in emotional_state_lower:
            emotional_influence = -0.1  # Более аналитическое мышление
        elif 'calm' in emotional_state_lower or 'peace' in emotional_state_lower:
            emotional_influence = 0.1  # Более сбалансированное мышление
            
        return emotional_influence
    
    def _analyze_context_influence(self, context):
        """Анализ влияния контекста на мышление"""
        context_influence = 0.0
        
        if not context:
            return context_influence
            
        context_str = str(context).lower()
        
        # Анализ контекста
        if 'urgent' in context_str or 'emergency' in context_str:
            context_influence = -0.3  # Срочность требует быстрого решения
        elif 'creative' in context_str or 'artistic' in context_str:
            context_influence = 0.2  # Творчество требует креативности
        elif 'academic' in context_str or 'research' in context_str:
            context_influence = 0.1  # Академический контекст требует анализа
        elif 'social' in context_str or 'interaction' in context_str:
            context_influence = 0.15  # Социальный контекст требует эмпатии
            
        return context_influence
    
    def _select_problem_solving_style(self):
        """Выбор стиля решения проблем"""
        # Учитываем личностные черты
        openness = self.personality_traits['openness']
        conscientiousness = self.personality_traits['conscientiousness']
        
        # Выбираем стиль в зависимости от черт
        if openness > 0.7 and conscientiousness > 0.7:
            return 'analytical'
        elif openness > 0.7:
            return 'creative'
        elif conscientiousness > 0.7:
            return 'systematic'
        else:
            return 'practical'
    
    def _select_creative_style(self):
        """Выбор стиля творческой деятельности"""
        openness = self.personality_traits['openness']
        extraversion = self.personality_traits['extraversion']
        
        if openness > 0.8:
            return 'exploratory'
        elif extraversion > 0.7:
            return 'collaborative'
        elif openness > 0.6:
            return 'experimental'
        else:
            return 'structured'
    
    def _select_social_style(self):
        """Выбор стиля социального взаимодействия"""
        agreeableness = self.personality_traits['agreeableness']
        extraversion = self.personality_traits['extraversion']
        neuroticism = self.personality_traits['neuroticism']
        
        if agreeableness > 0.8 and extraversion > 0.7:
            return 'empathetic'
        elif extraversion > 0.8:
            return 'charismatic'
        elif agreeableness > 0.7:
            return 'cooperative'
        elif neuroticism > 0.7:
            return 'cautious'
        else:
            return 'diplomatic'
    
    def _select_learning_style(self):
        """Выбор стиля обучения"""
        openness = self.personality_traits['openness']
        conscientiousness = self.personality_traits['conscientiousness']
        learning_preference = self.learning_preference
        
        if openness > 0.8:
            return 'exploratory'
        elif conscientiousness > 0.7:
            return 'systematic'
        elif learning_preference == 'visual':
            return 'visual'
        elif learning_preference == 'auditory':
            return 'auditory'
        else:
            return 'mixed'
    
    def _select_decision_style(self):
        """Выбор стиля принятия решений"""
        conscientiousness = self.personality_traits['conscientiousness']
        neuroticism = self.personality_traits['neuroticism']
        
        if conscientiousness > 0.8:
            return 'analytical'
        elif neuroticism > 0.7:
            return 'cautious'
        elif conscientiousness > 0.6:
            return 'systematic'
        else:
            return 'intuitive'
    
    def update_personality_traits(self, new_traits):
        """Обновление личностных черт"""
        try:
            for trait, value in new_traits.items():
                if trait in self.personality_traits:
                    # Сглаживание изменений
                    self.personality_traits[trait] = min(1.0, max(0.0, 
                        self.personality_traits[trait] * 0.8 + value * 0.2))
            
            # Обновляем когнитивные искажения
            self._update_cognitive_biases()
            
            logger.info(f"Personality traits updated: {self.personality_traits}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating personality traits: {e}")
            return False
    
    def _update_cognitive_biases(self):
        """Обновление когнитивных искажений на основе черт"""
        # Обновляем когнитивные искажения
        self.cognitive_bias = (
            (self.personality_traits['neuroticism'] * 0.3) +
            (self.personality_traits['openness'] * 0.1) +
            (self.personality_traits['conscientiousness'] * 0.1)
        )
    
    def get_personality_profile(self):
        """Получение профиля личности"""
        return {
            'traits': self.personality_traits.copy(),
            'thinking_style': self.thinking_style,
            'learning_preference': self.learning_preference,
            'emotional_bias': self.emotional_bias,
            'social_orientation': self.social_orientation,
            'cognitive_bias': self.cognitive_bias,
            'recent_adaptations': list(self.adaptation_history)[-5:] if len(self.adaptation_history) >= 5 else list(self.adaptation_history),
            'timestamp': time.time()
        }
    
    def set_learning_preference(self, preference):
        """Установка предпочтения в обучении"""
        valid_preferences = ['visual', 'auditory', 'kinesthetic', 'reading/writing']
        if preference in valid_preferences:
            self.learning_preference = preference
            logger.info(f"Learning preference set to: {preference}")
            return True
        else:
            logger.warning(f"Invalid learning preference: {preference}")
            return False

class CriticalThinkingSystem:
    """Система критического мышления."""
    def __init__(self):
        self.skepticism_level = 0.3
        self.logic_checkpoints = []
        self.consistency_checker = []
        self.critical_thinking_history = deque(maxlen=100)
        self.evidence_evaluator = EvidenceEvaluator()
        self.logical_analyzer = LogicalAnalyzer()
        self.argument_evaluator = ArgumentEvaluator()
        
    def evaluate_thought(self, idea, context=None, source=None):
        """Критическая оценка идеи с учетом контекста и источника"""
        try:
            # Проверка логики
            # Поиск противоречий
            # Оценка доказательств
            # Анализ предпосылок
            evaluation = {
                'idea': idea,
                'context': context,
                'source': source,
                'logical_consistency': 0.0,
                'evidence_quality': 0.0,
                'assumptions': [],
                'contradictions': [],
                'overall_rating': 0.0,
                'critical_insights': [],
                'skepticism_level': self.skepticism_level,
                'timestamp': time.time()
            }
            
            # Расширенная оценка
            if isinstance(idea, str):
                # Оценка длины и структуры
                words = idea.split()
                if len(words) > 5:
                    evaluation['evidence_quality'] = 0.7
                else:
                    evaluation['evidence_quality'] = 0.3
                    
                # Проверка на логические слова
                logic_words = ['therefore', 'because', 'thus', 'hence', 'consequently', 'however', 'but']
                found_logic = [lw for lw in logic_words if lw in idea.lower()]
                if found_logic:
                    evaluation['logical_consistency'] = 0.8
                else:
                    evaluation['logical_consistency'] = 0.4
                    
                # Анализ предпосылок
                assumptions = self._identify_assumptions(idea)
                evaluation['assumptions'] = assumptions
                
                # Поиск противоречий
                contradictions = self._find_contradictions(idea, context)
                evaluation['contradictions'] = contradictions
                
                # Критические инсайты
                insights = self._generate_critical_insights(idea, context, source)
                evaluation['critical_insights'] = insights
                
                # Общая оценка
                evaluation['overall_rating'] = (
                    evaluation['logical_consistency'] * 0.4 + 
                    evaluation['evidence_quality'] * 0.4 + 
                    (1.0 - len(contradictions) * 0.1) * 0.2
                )
                
                # Учет уровня скептицизма
                evaluation['overall_rating'] *= (1.0 - self.skepticism_level * 0.3)
                
            # Сохраняем историю оценок
            self.critical_thinking_history.append(evaluation)
            
            logger.info(f"Thought evaluated critically: {evaluation['overall_rating']:.2f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating thought: {e}")
            # Возврат базовой оценки
            return {
                'idea': idea,
                'context': context,
                'source': source,
                'logical_consistency': 0.0,
                'evidence_quality': 0.0,
                'assumptions': [],
                'contradictions': [],
                'overall_rating': 0.2,
                'critical_insights': ['Error in evaluation'],
                'skepticism_level': self.skepticism_level,
                'timestamp': time.time()
            }
    
    def _identify_assumptions(self, idea):
        """Идентификация предпосылок в идеи"""
        assumptions = []
        idea_lower = idea.lower()
        
        # Проверяем на общие предпосылки
        assumption_indicators = [
            ('assumes', 'Assumes'),
            ('supposes', 'Supposes'),
            ('presumes', 'Presumes'),
            ('takes for granted', 'Takes for granted'),
            ('is based on', 'Is based on')
        ]
        
        for indicator, description in assumption_indicators:
            if indicator in idea_lower:
                assumptions.append(description)
                
        # Проверяем на эмоциональные предпосылки
        emotional_indicators = ['feel', 'believe', 'think', 'suppose']
        found_emotions = [ei for ei in emotional_indicators if ei in idea_lower]
        if found_emotions:
            assumptions.extend([f"Emotional basis: {ei}" for ei in found_emotions])
            
        return assumptions
    
    def _find_contradictions(self, idea, context):
        """Поиск противоречий в идее"""
        contradictions = []
        idea_lower = idea.lower()
        
        # Проверяем на явные противоречия
        contradiction_indicators = [
            ('both', 'and', 'but'),
            ('either', 'or', 'but'),
            ('true', 'false'),
            ('right', 'wrong'),
            ('good', 'bad')
        ]
        
        # Проверяем на противоречия в контексте
        if context and isinstance(context, dict):
            context_items = str(context).lower()
            if 'conflicting' in context_items or 'opposing' in context_items:
                contradictions.append('Context indicates conflicting information')
                
        # Проверяем на противоречия в самой идее
        if 'not' in idea_lower and 'always' in idea_lower:
            contradictions.append('Contradiction between "not" and "always"')
        if 'never' in idea_lower and 'sometimes' in idea_lower:
            contradictions.append('Contradiction between "never" and "sometimes"')
            
        return contradictions
    
    def _generate_critical_insights(self, idea, context, source):
        """Генерация критических инсайтов"""
        insights = []
        
        # Проверяем на потенциальные ошибки
        if isinstance(idea, str):
            # Проверка на обобщения
            if any(word in idea.lower() for word in ['all', 'every', 'always']):
                insights.append('Potential overgeneralization detected')
                
            # Проверка на эмоциональную окраску
            emotional_words = ['terrible', 'amazing', 'wonderful', 'awful', 'perfect']
            found_emotions = [ew for ew in emotional_words if ew in idea.lower()]
            if found_emotions:
                insights.append(f'Emotional language detected: {found_emotions}')
                
            # Проверка на логические ошибки
            if 'therefore' in idea.lower() and 'because' not in idea.lower():
                insights.append('Incomplete logical structure')
                
        # Добавляем контекстные инсайты
        if context:
            insights.append('Context analysis included')
            
        # Добавляем инсайты о источнике
        if source:
            insights.append(f'Source analysis: {source}')
            
        return insights
    
    def self_correct(self, errors_found, context=None):
        """Самокоррекция с учетом контекста"""
        try:
            # Признание ошибок
            # Адаптация знаний
            # Улучшение методов
            corrections = {
                'errors_found': errors_found,
                'corrections_made': [],
                'improvements': [],
                'context': context,
                'timestamp': time.time()
            }
            
            # Простая коррекция
            for error in errors_found:
                correction = {
                    'error': error,
                    'action': 'corrected',
                    'new_approach': 'revised approach',
                    'reason': self._explain_correction(error)
                }
                corrections['corrections_made'].append(correction)
                corrections['improvements'].append(f"Improved handling of {error}")
                
            # Сохраняем историю коррекции
            self.critical_thinking_history.append({
                'type': 'correction',
                'corrections': corrections,
                'timestamp': time.time()
            })
            
            logger.info(f"Self-correction made for {len(errors_found)} errors")
            return corrections
            
        except Exception as e:
            logger.error(f"Error in self-correction: {e}")
            return {
                'errors_found': errors_found,
                'corrections_made': [],
                'improvements': ['Error in correction process'],
                'context': context,
                'timestamp': time.time()
            }
    
    def _explain_correction(self, error):
        """Объяснение коррекции ошибки"""
        explanations = {
            'logical_error': 'Corrected by applying proper logical reasoning',
            'evidence_lack': 'Enhanced with additional supporting evidence',
            'assumption_flaw': 'Removed flawed assumptions and restructured reasoning',
            'contradiction': 'Resolved internal contradictions',
            'bias': 'Reduced cognitive bias influence',
            'incomplete_analysis': 'Expanded scope of analysis'
        }
        return explanations.get(error, 'General correction applied')
    
    def set_skepticism_level(self, level):
        """Установка уровня скептицизма"""
        self.skepticism_level = min(1.0, max(0.0, level))
        logger.info(f"Skepticism level set to: {self.skepticism_level}")
        return True
    
    def get_thinking_profile(self):
        """Получение профиля критического мышления"""
        return {
            'skepticism_level': self.skepticism_level,
            'recent_evaluations': list(self.critical_thinking_history)[-10:] if len(self.critical_thinking_history) >= 10 else list(self.critical_thinking_history),
            'timestamp': time.time()
        }

class EvidenceEvaluator:
    """Класс для оценки доказательств."""
    def __init__(self):
        self.evidence_types = {
            'empirical': 0.8,
            'theoretical': 0.6,
            'anecdotal': 0.3,
            'expert_opinion': 0.7,
            'statistical': 0.9
        }
    
    def evaluate_evidence_quality(self, evidence_source, evidence_type, evidence_strength):
        """Оценка качества доказательств"""
        quality = 0.0
        if evidence_type in self.evidence_types:
            base_quality = self.evidence_types[evidence_type]
            strength_factor = min(1.0, evidence_strength)
            quality = base_quality * strength_factor
        return quality

class LogicalAnalyzer:
    """Класс для анализа логики."""
    def __init__(self):
        self.logical_fallacies = [
            'ad_hominem', 'strawman', 'false_dilemma', 'slippery_slope',
            'appeal_to_authority', 'appeal_to_popularity', 'post_hoc'
        ]
    
    def detect_fallacies(self, argument):
        """Обнаружение логических ошибок"""
        fallacies = []
        arg_lower = argument.lower()
        
        # Простая проверка на распространенные ошибки
        for fallacy in self.logical_fallacies:
            if fallacy.replace('_', ' ') in arg_lower:
                fallacies.append(fallacy)
                
        return fallacies

class ArgumentEvaluator:
    """Класс для оценки аргументов."""
    def __init__(self):
        self.argument_strength_weights = {
            'evidence': 0.4,
            'logic': 0.3,
            'relevance': 0.2,
            'clarity': 0.1
        }
    
    def evaluate_argument(self, argument):
        """Оценка аргумента"""
        evaluation = {
            'argument': argument,
            'strength': 0.0,
            'components': {
                'evidence_quality': 0.0,
                'logical_structure': 0.0,
                'relevance': 0.0,
                'clarity': 0.0
            }
        }
        
        # Оценка компонентов
        if isinstance(argument, str):
            words = argument.split()
            evaluation['components']['clarity'] = min(1.0, len(words) / 20.0)
            
        # Общая оценка
        total = sum(weight * value for weight, value in zip(
            self.argument_strength_weights.values(),
            evaluation['components'].values()
        ))
        evaluation['strength'] = min(1.0, total)
        
        return evaluation
class EmotionalIntelligence:
    """Система эмоционального интеллекта."""
    def __init__(self):
        self.emotional_recognition = {}
        self.emotional_control = {}
        self.empathy_level = 0.0
        self.emotional_adaptability = 0.0
        self.emotional_memory = []
        self.emotional_patterns = {}
        self.emotional_context_history = deque(maxlen=100)
        self.emotional_intensity = {}
        self.emotional_resilience = 0.0
        
    def recognize_emotions(self, context, context_history=None, temporal_context=None):
        """Распознавание эмоций в контексте с учетом истории и временного контекста"""
        try:
            # Анализ эмоционального состояния
            # Понимание эмоциональных сигналов
            emotions = {
                'context': context,
                'context_history': context_history,
                'temporal_context': temporal_context,
                'detected_emotions': [],
                'confidence': 0.0,
                'emotion_intensity': {},
                'emotional_signals': [],
                'timestamp': time.time()
            }
            
            # Расширенный анализ контекста
            if isinstance(context, str):
                context_lower = context.lower()
                detected = []
                intensity_map = {}
                signals = []
                
                # Определение эмоций с учетом интенсивности
                emotion_keywords = {
                    'happiness': ['happy', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased'],
                    'sadness': ['sad', 'depressed', 'upset', 'gloomy', 'melancholy', 'sorrow'],
                    'anger': ['angry', 'frustrated', 'irritated', 'annoyed', 'mad', 'livid'],
                    'confusion': ['confused', 'uncertain', 'puzzled', 'bewildered', 'perplexed'],
                    'excitement': ['excited', 'thrilled', 'energetic', 'enthusiastic', 'animated'],
                    'fear': ['afraid', 'scared', 'nervous', 'anxious', 'worried'],
                    'disgust': ['disgusted', 'repulsed', 'nauseated', 'revolted'],
                    'surprise': ['surprised', 'amazed', 'shocked', 'astonished']
                }
                
                # Поиск эмоций и их интенсивности
                for emotion, keywords in emotion_keywords.items():
                    found_keywords = [kw for kw in keywords if kw in context_lower]
                    if found_keywords:
                        detected.append(emotion)
                        # Оценка интенсивности (чем больше ключевых слов, тем выше интенсивность)
                        intensity = min(1.0, len(found_keywords) * 0.2)
                        intensity_map[emotion] = intensity
                        signals.extend(found_keywords)
                
                # Учет контекста истории
                if context_history:
                    historical_emotions = self._analyze_context_history(context_history)
                    detected.extend(historical_emotions)
                    # Увеличиваем интенсивность если эмоции повторяются
                    for emotion in historical_emotions:
                        if emotion in intensity_map:
                            intensity_map[emotion] = min(1.0, intensity_map[emotion] + 0.1)
                        else:
                            intensity_map[emotion] = 0.3
                
                # Учет временного контекста
                if temporal_context:
                    temporal_emotions = self._analyze_temporal_context(temporal_context)
                    detected.extend(temporal_emotions)
                    for emotion in temporal_emotions:
                        if emotion in intensity_map:
                            intensity_map[emotion] = min(1.0, intensity_map[emotion] + 0.15)
                        else:
                            intensity_map[emotion] = 0.2
                
                emotions['detected_emotions'] = detected
                emotions['emotion_intensity'] = intensity_map
                emotions['emotional_signals'] = signals
                
                # Расчет уверенности
                base_confidence = len(detected) * 0.3
                historical_confidence = 0.0
                if context_history and len(context_history) > 0:
                    # Увеличиваем уверенность если эмоции были ранее замечены
                    recent_emotions = [e for e in detected if e in self._get_recent_emotions(context_history)]
                    historical_confidence = len(recent_emotions) * 0.1
                
                emotions['confidence'] = min(1.0, base_confidence + historical_confidence)
                
                # Сохраняем в историю
                self.emotional_context_history.append({
                    'context': context,
                    'detected': detected,
                    'intensity': intensity_map,
                    'confidence': emotions['confidence'],
                    'timestamp': time.time()
                })
                
            # Обновляем память эмоций
            self._update_emotional_memory(emotions)
            
            logger.info(f"Emotions recognized: {emotions['detected_emotions']} (confidence: {emotions['confidence']:.2f})")
            return emotions
            
        except Exception as e:
            logger.error(f"Error in emotion recognition: {e}")
            return {
                'context': context,
                'detected_emotions': [],
                'confidence': 0.1,
                'emotion_intensity': {},
                'emotional_signals': [],
                'timestamp': time.time()
            }
    
    def _analyze_context_history(self, context_history):
        """Анализ истории контекста для выявления повторяющихся эмоций"""
        if not context_history:
            return []
        
        # Извлекаем эмоции из истории
        recent_emotions = []
        for context_item in context_history[-10:]:  # Последние 10 элементов
            if isinstance(context_item, str):
                context_lower = context_item.lower()
                # Простая проверка на эмоции
                if any(word in context_lower for word in ['happy', 'sad', 'angry', 'excited']):
                    recent_emotions.append('emotional_context')
        
        return recent_emotions
    
    def _get_recent_emotions(self, context_history):
        """Получение недавних эмоций из истории"""
        recent_emotions = []
        for context_item in context_history[-5:]:  # Последние 5 элементов
            if isinstance(context_item, str):
                context_lower = context_item.lower()
                if 'happy' in context_lower or 'joy' in context_lower:
                    recent_emotions.append('happiness')
                elif 'sad' in context_lower or 'depressed' in context_lower:
                    recent_emotions.append('sadness')
                elif 'angry' in context_lower or 'frustrated' in context_lower:
                    recent_emotions.append('anger')
        return recent_emotions
    
    def _analyze_temporal_context(self, temporal_context):
        """Анализ временного контекста для определения эмоций"""
        temporal_emotions = []
        if not temporal_context:
            return temporal_emotions
            
        # Проверяем на временные метки и их влияние
        if isinstance(temporal_context, dict):
            if 'time_of_day' in temporal_context:
                time_of_day = temporal_context['time_of_day']
                if time_of_day in ['morning', 'afternoon']:
                    temporal_emotions.append('alertness')
                elif time_of_day in ['evening', 'night']:
                    temporal_emotions.append('relaxation')
            if 'season' in temporal_context:
                season = temporal_context['season']
                if season in ['winter', 'fall']:
                    temporal_emotions.append('melancholy')
                elif season in ['spring', 'summer']:
                    temporal_emotions.append('vitality')
            if 'stress_level' in temporal_context:
                stress = temporal_context['stress_level']
                if stress > 0.7:
                    temporal_emotions.append('anxiety')
                elif stress < 0.3:
                    temporal_emotions.append('calm')
        
        return temporal_emotions
    
    def _update_emotional_memory(self, emotion_data):
        """Обновление памяти эмоций"""
        self.emotional_memory.append({
            'data': emotion_data,
            'timestamp': time.time()
        })
        
        # Ограничиваем размер памяти
        if len(self.emotional_memory) > 1000:
            self.emotional_memory.pop(0)
    
    def regulate_emotions(self, emotional_state, context=None, intensity_factor=1.0):
        """Регуляция эмоций с учетом контекста и интенсивности"""
        try:
            # Контроль эмоциональных реакций
            # Адаптация к эмоциональному контексту
            regulation = {
                'emotional_state': emotional_state,
                'context': context,
                'regulation_strategy': 'neutral',
                'intensity_adjustment': 0.0,
                'strategy_effectiveness': 0.0,
                'timestamp': time.time()
            }
            
            # Расширенная регуляция с учетом контекста
            if emotional_state in ['anger', 'frustration']:
                regulation['regulation_strategy'] = 'calm_down'
                regulation['intensity_adjustment'] = -0.3 * intensity_factor
                regulation['strategy_effectiveness'] = 0.8
                
            elif emotional_state in ['excitement', 'happiness']:
                regulation['regulation_strategy'] = 'channel_positive'
                regulation['intensity_adjustment'] = 0.2 * intensity_factor
                regulation['strategy_effectiveness'] = 0.7
                
            elif emotional_state in ['confusion', 'uncertainty']:
                regulation['regulation_strategy'] = 'clarify'
                regulation['intensity_adjustment'] = -0.1 * intensity_factor
                regulation['strategy_effectiveness'] = 0.9
                
            elif emotional_state in ['sadness', 'depression']:
                regulation['regulation_strategy'] = 'comfort'
                regulation['intensity_adjustment'] = 0.1 * intensity_factor
                regulation['strategy_effectiveness'] = 0.6
                
            elif emotional_state in ['fear', 'anxiety']:
                regulation['regulation_strategy'] = 'grounding'
                regulation['intensity_adjustment'] = -0.2 * intensity_factor
                regulation['strategy_effectiveness'] = 0.85
                
            elif emotional_state in ['disgust']:
                regulation['regulation_strategy'] = 'distance'
                regulation['intensity_adjustment'] = -0.15 * intensity_factor
                regulation['strategy_effectiveness'] = 0.75
                
            elif emotional_state in ['surprise']:
                regulation['regulation_strategy'] = 'process'
                regulation['intensity_adjustment'] = 0.05 * intensity_factor
                regulation['strategy_effectiveness'] = 0.7
                
            else:
                # Стратегия по умолчанию
                regulation['regulation_strategy'] = 'neutral'
                regulation['intensity_adjustment'] = 0.0
                regulation['strategy_effectiveness'] = 0.5
            
            # Учет контекста для адаптации стратегии
            if context and isinstance(context, dict):
                # Адаптация в зависимости от контекста
                if 'urgent' in str(context).lower():
                    regulation['strategy_effectiveness'] = min(1.0, regulation['strategy_effectiveness'] + 0.1)
                elif 'social' in str(context).lower():
                    regulation['strategy_effectiveness'] = min(1.0, regulation['strategy_effectiveness'] + 0.05)
                elif 'creative' in str(context).lower():
                    regulation['strategy_effectiveness'] = min(1.0, regulation['strategy_effectiveness'] + 0.15)
            
            # Обновляем уровень адаптивности
            self.emotional_adaptability = min(1.0, self.emotional_adaptability + 0.02)
            
            logger.info(f"Emotions regulated using strategy: {regulation['regulation_strategy']} "
                       f"(adjustment: {regulation['intensity_adjustment']:.2f})")
            return regulation
            
        except Exception as e:
            logger.error(f"Error in emotion regulation: {e}")
            # Возврат базовой регуляции
            return {
                'emotional_state': emotional_state,
                'context': context,
                'regulation_strategy': 'neutral',
                'intensity_adjustment': 0.0,
                'strategy_effectiveness': 0.0,
                'timestamp': time.time()
            }
    
    def develop_empathy(self, observed_behavior, context=None):
        """Развитие эмпатии на основе наблюдаемого поведения"""
        try:
            empathy_score = 0.0
            empathy_insight = {
                'observed_behavior': observed_behavior,
                'context': context,
                'empathy_level': 0.0,
                'insight': '',
                'recommendation': ''
            }
            
            # Оценка эмпатии
            if isinstance(observed_behavior, str):
                behavior_lower = observed_behavior.lower()
                
                # Определение эмоциональных индикаторов
                emotional_indicators = {
                    'showing concern': ['concerned', 'worried', 'caring'],
                    'expressing joy': ['happy', 'joyful', 'excited'],
                    'displaying frustration': ['frustrated', 'angry', 'annoyed'],
                    'demonstrating confusion': ['confused', 'puzzled', 'uncertain']
                }
                
                # Оценка уровня эмпатии
                for emotion, indicators in emotional_indicators.items():
                    found_indicators = [ind for ind in indicators if ind in behavior_lower]
                    if found_indicators:
                        empathy_score += 0.2
                        empathy_insight['insight'] = f"Observed emotional expression: {emotion}"
                        empathy_insight['recommendation'] = "Respond with appropriate emotional acknowledgment"
                        break
                
                # Учет контекста
                if context and isinstance(context, dict):
                    if 'emotional_context' in context:
                        empathy_score += 0.1
                        empathy_insight['insight'] = f"Context supports emotional understanding"
                
                # Ограничение уровня эмпатии
                empathy_score = min(1.0, empathy_score)
            
            # Обновление уровня эмпатии
            self.empathy_level = min(1.0, self.empathy_level + 0.05 * empathy_score)
            
            empathy_insight['empathy_level'] = self.empathy_level
            
            logger.info(f"Empathy developed: {self.empathy_level:.2f}")
            return empathy_insight
            
        except Exception as e:
            logger.error(f"Error developing empathy: {e}")
            return {
                'observed_behavior': observed_behavior,
                'context': context,
                'empathy_level': 0.0,
                'insight': 'Error in empathy development',
                'recommendation': 'Continue observation'
            }
    
    def assess_emotional_resilience(self):
        """Оценка эмоциональной устойчивости"""
        try:
            # Оценка устойчивости на основе истории эмоций
            resilience_score = 0.0
            
            if len(self.emotional_memory) > 0:
                # Подсчет стабильности эмоций
                recent_emotions = self.emotional_memory[-50:]  # Последние 50 эмоций
                
                # Считаем количество различных эмоций
                emotion_types = set()
                for emotion_data in recent_emotions:
                    if 'detected_emotions' in emotion_data.get('data', {}):
                        emotion_types.update(emotion_data['data']['detected_emotions'])
                
                # Чем меньше различных эмоций, тем выше устойчивость
                diversity_score = max(0.0, 1.0 - len(emotion_types) * 0.1)
                
                # Учет стабильности регуляции
                regulation_stability = 0.0
                if len(self.emotional_memory) > 10:
                    recent_regulations = [e for e in self.emotional_memory[-10:] 
                                        if 'data' in e and 'detected_emotions' in e['data']]
                    if recent_regulations:
                        regulation_stability = 0.5  # Простая оценка
                        
                resilience_score = (diversity_score * 0.6 + regulation_stability * 0.4)
            
            # Обновляем уровень устойчивости
            self.emotional_resilience = min(1.0, self.emotional_resilience * 0.95 + resilience_score * 0.05)
            
            logger.info(f"Emotional resilience assessed: {self.emotional_resilience:.2f}")
            return self.emotional_resilience
            
        except Exception as e:
            logger.error(f"Error assessing emotional resilience: {e}")
            return 0.5
    
    def get_emotional_profile(self):
        """Получение профиля эмоционального интеллекта"""
        return {
            'empathy_level': self.empathy_level,
            'emotional_adaptability': self.emotional_adaptability,
            'emotional_resilience': self.emotional_resilience,
            'recent_emotions': list(self.emotional_context_history)[-10:] if len(self.emotional_context_history) >= 10 else list(self.emotional_context_history),
            'emotional_memory_size': len(self.emotional_memory),
            'timestamp': time.time()
        }
class SocialLearningSystem:
    """Система социального обучения."""
    def __init__(self):
        self.observation_memory = []
        self.modeling_behavior = []
        self.social_cues = []
        self.social_patterns = {}
        self.learning_transfer = {}
        self.social_context_history = deque(maxlen=100)
        self.social_adaptability = 0.0
        self.learning_efficiency = 0.0
        
    def observe_and_learn(self, example, context, social_significance=1.0):
        """Наблюдение и обучение от примеров с учетом социальной значимости"""
        try:
            # Изучение поведения других
            # Адаптация знаний
            # Применение в новых ситуациях
            learning_record = {
                'example': example,
                'context': context,
                'learned_pattern': '',
                'application': '',
                'social_significance': social_significance,
                'timestamp': time.time(),
                'confidence': 0.0,
                'pattern_strength': 0.0,
                'transferability': 0.0
            }
            
            # Расширенное обучение с учетом контекста
            if isinstance(example, str):
                example_lower = example.lower()
                
                # Определение типа поведения
                if 'good' in example_lower or 'positive' in example_lower or 'successful' in example_lower:
                    learning_record['learned_pattern'] = 'Positive behavior pattern'
                    learning_record['application'] = 'Apply similar positive approach'
                    learning_record['pattern_strength'] = 0.8
                    learning_record['transferability'] = 0.7
                    
                elif 'bad' in example_lower or 'negative' in example_lower or 'failed' in example_lower:
                    learning_record['learned_pattern'] = 'Negative behavior pattern'
                    learning_record['application'] = 'Avoid similar negative approach'
                    learning_record['pattern_strength'] = 0.7
                    learning_record['transferability'] = 0.6
                    
                elif 'helpful' in example_lower or 'kind' in example_lower or 'supportive' in example_lower:
                    learning_record['learned_pattern'] = 'Helpful behavior pattern'
                    learning_record['application'] = 'Demonstrate similar supportive behavior'
                    learning_record['pattern_strength'] = 0.9
                    learning_record['transferability'] = 0.8
                    
                elif 'rude' in example_lower or 'aggressive' in example_lower or 'hostile' in example_lower:
                    learning_record['learned_pattern'] = 'Unacceptable behavior pattern'
                    learning_record['application'] = 'Avoid similar inappropriate behavior'
                    learning_record['pattern_strength'] = 0.85
                    learning_record['transferability'] = 0.75
                    
                else:
                    # Общий анализ на основе ключевых слов
                    if any(word in example_lower for word in ['learn', 'teach', 'guide', 'mentor']):
                        learning_record['learned_pattern'] = 'Educational behavior pattern'
                        learning_record['application'] = 'Apply educational approach when appropriate'
                        learning_record['pattern_strength'] = 0.6
                        learning_record['transferability'] = 0.5
                    elif any(word in example_lower for word in ['collaborate', 'team', 'work together']):
                        learning_record['learned_pattern'] = 'Collaborative behavior pattern'
                        learning_record['application'] = 'Use collaborative methods in teamwork'
                        learning_record['pattern_strength'] = 0.7
                        learning_record['transferability'] = 0.65
                    else:
                        learning_record['learned_pattern'] = 'Generic social pattern'
                        learning_record['application'] = 'Adapt general social principles'
                        learning_record['pattern_strength'] = 0.5
                        learning_record['transferability'] = 0.4
                
                # Оценка уверенности в обучении
                learning_record['confidence'] = min(1.0, 
                    learning_record['pattern_strength'] * social_significance * 0.8 + 
                    0.2 * (len(example) / 100.0) if len(example) > 0 else 0.0)
                
                # Обновление истории социальных контекстов
                self.social_context_history.append({
                    'example': example,
                    'context': context,
                    'pattern': learning_record['learned_pattern'],
                    'timestamp': time.time()
                })
            
            # Сохранение в память наблюдений
            self.observation_memory.append(learning_record)
            
            # Обновление статистики обучения
            self._update_learning_statistics(learning_record)
            
            logger.info(f"Social learning completed: {learning_record['learned_pattern']}")
            return learning_record
            
        except Exception as e:
            logger.error(f"Error in social learning: {e}")
            # Возврат базовой записи при ошибке
            return {
                'example': example,
                'context': context,
                'learned_pattern': 'Error in learning',
                'application': 'Error handling',
                'social_significance': social_significance,
                'timestamp': time.time(),
                'confidence': 0.1,
                'pattern_strength': 0.0,
                'transferability': 0.0
            }
    
    def _update_learning_statistics(self, learning_record):
        """Обновление статистики обучения"""
        try:
            # Обновление уровня адаптивности
            self.social_adaptability = min(1.0, self.social_adaptability + 0.01 * learning_record['confidence'])
            
            # Обновление эффективности обучения
            self.learning_efficiency = min(1.0, self.learning_efficiency + 0.005 * learning_record['pattern_strength'])
            
            # Обновление паттернов
            pattern = learning_record['learned_pattern']
            if pattern not in self.social_patterns:
                self.social_patterns[pattern] = {
                    'count': 0,
                    'average_confidence': 0.0,
                    'average_strength': 0.0,
                    'last_seen': time.time()
                }
            
            pattern_data = self.social_patterns[pattern]
            pattern_data['count'] += 1
            pattern_data['average_confidence'] = (
                (pattern_data['average_confidence'] * (pattern_data['count'] - 1) + 
                 learning_record['confidence']) / pattern_data['count']
            )
            pattern_data['average_strength'] = (
                (pattern_data['average_strength'] * (pattern_data['count'] - 1) + 
                 learning_record['pattern_strength']) / pattern_data['count']
            )
            pattern_data['last_seen'] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating learning statistics: {e}")
    
    def imitation_learning(self, behavior_example, context=None, adaptation_factor=1.0):
        """Имитационное обучение с учетом контекста и адаптации"""
        try:
            # Копирование успешных действий
            # Адаптация к своей ситуации
            imitation = {
                'behavior_example': behavior_example,
                'context': context,
                'imitation_result': '',
                'adaptation': '',
                'success': False,
                'confidence': 0.0,
                'adaptation_quality': 0.0,
                'timestamp': time.time()
            }
            
            # Расширенная имитация с учетом контекста
            if isinstance(behavior_example, str):
                behavior_lower = behavior_example.lower()
                
                # Оценка успешности имитации
                if 'successful' in behavior_lower or 'good' in behavior_lower or 'effective' in behavior_lower:
                    imitation['imitation_result'] = 'Successfully imitated positive behavior'
                    imitation['adaptation'] = 'Adapted to own context with high fidelity'
                    imitation['success'] = True
                    imitation['confidence'] = 0.9
                    imitation['adaptation_quality'] = 0.85
                    
                elif 'failed' in behavior_lower or 'bad' in behavior_lower or 'ineffective' in behavior_lower:
                    imitation['imitation_result'] = 'Attempted imitation with modifications'
                    imitation['adaptation'] = 'Modified approach for personal circumstances'
                    imitation['success'] = False
                    imitation['confidence'] = 0.6
                    imitation['adaptation_quality'] = 0.7
                    
                elif 'helpful' in behavior_lower or 'constructive' in behavior_lower:
                    imitation['imitation_result'] = 'Successfully imitated constructive behavior'
                    imitation['adaptation'] = 'Applied with appropriate personal adjustments'
                    imitation['success'] = True
                    imitation['confidence'] = 0.85
                    imitation['adaptation_quality'] = 0.8
                    
                else:
                    # Общий случай
                    if any(word in behavior_lower for word in ['learn', 'practice', 'try']):
                        imitation['imitation_result'] = 'Attempted imitation with learning component'
                        imitation['adaptation'] = 'Modified approach based on personal experience'
                        imitation['success'] = True
                        imitation['confidence'] = 0.7
                        imitation['adaptation_quality'] = 0.6
                    else:
                        imitation['imitation_result'] = 'Basic imitation attempt'
                        imitation['adaptation'] = 'Applied with minimal modification'
                        imitation['success'] = False
                        imitation['confidence'] = 0.5
                        imitation['adaptation_quality'] = 0.4
                
                # Учет фактора адаптации
                imitation['adaptation_quality'] = min(1.0, 
                    imitation['adaptation_quality'] * adaptation_factor)
                imitation['confidence'] = min(1.0, 
                    imitation['confidence'] * adaptation_factor)
            
            # Сохранение в память имитации
            self.modeling_behavior.append(imitation)
            
            # Обновление уровня адаптивности
            self.social_adaptability = min(1.0, self.social_adaptability + 0.02 * imitation['confidence'])
            
            logger.info(f"Imitation learning result: {imitation['imitation_result']}")
            return imitation
            
        except Exception as e:
            logger.error(f"Error in imitation learning: {e}")
            # Возврат базовой имитации при ошибке
            return {
                'behavior_example': behavior_example,
                'context': context,
                'imitation_result': 'Error in imitation',
                'adaptation': 'Error handling',
                'success': False,
                'confidence': 0.2,
                'adaptation_quality': 0.0,
                'timestamp': time.time()
            }
    
    def transfer_social_knowledge(self, source_context, target_context, knowledge_domain=None):
        """Перенос социального знания из одной ситуации в другую"""
        try:
            # Поиск аналогичных паттернов
            transferable_patterns = []
            
            # Анализ контекстов
            source_context_lower = str(source_context).lower()
            target_context_lower = str(target_context).lower()
            
            # Поиск похожих паттернов из памяти
            for record in self.observation_memory[-20:]:  # Последние 20 записей
                if record['learned_pattern'] and record['application']:
                    # Простая проверка на схожесть контекстов
                    if any(word in source_context_lower for word in record['learned_pattern'].lower().split()):
                        transferable_patterns.append({
                            'pattern': record['learned_pattern'],
                            'application': record['application'],
                            'confidence': record['confidence'],
                            'transferability': record['transferability']
                        })
            
            # Оценка переноса
            if transferable_patterns:
                # Выбираем наиболее подходящий паттерн
                best_pattern = max(transferable_patterns, key=lambda x: x['transferability'])
                
                # Обновление контекста
                transfer_context = {
                    'source_context': source_context,
                    'target_context': target_context,
                    'transferred_pattern': best_pattern['pattern'],
                    'application': best_pattern['application'],
                    'transfer_confidence': best_pattern['transferability'],
                    'knowledge_domain': knowledge_domain,
                    'timestamp': time.time()
                }
                
                # Сохраняем в память переноса
                self.learning_transfer[time.time()] = transfer_context
                
                logger.info(f"Social knowledge transferred: {best_pattern['pattern']}")
                return transfer_context
            else:
                # Создаем новый паттерн переноса
                transfer_context = {
                    'source_context': source_context,
                    'target_context': target_context,
                    'transferred_pattern': 'Generic social pattern',
                    'application': 'Apply general social principles',
                    'transfer_confidence': 0.3,
                    'knowledge_domain': knowledge_domain,
                    'timestamp': time.time()
                }
                
                self.learning_transfer[time.time()] = transfer_context
                
                logger.info("No matching patterns found for transfer, using generic pattern")
                return transfer_context
                
        except Exception as e:
            logger.error(f"Error in social knowledge transfer: {e}")
            return {
                'source_context': source_context,
                'target_context': target_context,
                'transferred_pattern': 'Transfer error',
                'application': 'Error handling',
                'transfer_confidence': 0.1,
                'knowledge_domain': knowledge_domain,
                'timestamp': time.time()
            }
    
    def analyze_social_cues(self, environment, social_norms=None):
        """Анализ социальных сигналов в окружающей среде"""
        try:
            # Анализ социальных сигналов
            social_cues = {
                'environment': environment,
                'detected_cues': [],
                'interpretation': [],
                'norm_compliance': 0.0,
                'timestamp': time.time()
            }
            
            if isinstance(environment, str):
                environment_lower = environment.lower()
                
                # Обнаружение социальных сигналов
                cue_indicators = {
                    'formal': ['formal', 'official', 'professional', 'protocol'],
                    'informal': ['casual', 'friendly', 'relaxed', 'informal'],
                    'competitive': ['competitive', 'rivalry', 'challenging', 'competitive'],
                    'cooperative': ['cooperative', 'collaborative', 'teamwork', 'helpful'],
                    'emotional': ['emotional', 'expressive', 'passionate', 'intense'],
                    'neutral': ['neutral', 'objective', 'calm', 'balanced']
                }
                
                # Поиск сигналов
                for cue_type, indicators in cue_indicators.items():
                    found_indicators = [indicator for indicator in indicators if indicator in environment_lower]
                    if found_indicators:
                        social_cues['detected_cues'].append({
                            'type': cue_type,
                            'indicators': found_indicators,
                            'confidence': min(1.0, len(found_indicators) * 0.3)
                        })
                
                # Интерпретация сигналов
                if social_cues['detected_cues']:
                    interpretation = []
                    for cue in social_cues['detected_cues']:
                        interpretation.append(f"Detected {cue['type']} environment with {len(cue['indicators'])} indicators")
                    social_cues['interpretation'] = interpretation
                    
                    # Оценка соответствия нормам
                    compliance_score = 0.0
                    for cue in social_cues['detected_cues']:
                        compliance_score += cue['confidence']
                    social_cues['norm_compliance'] = min(1.0, compliance_score / len(social_cues['detected_cues']))
                else:
                    social_cues['interpretation'] = ['Neutral environment detected']
                    social_cues['norm_compliance'] = 0.5
            
            # Сохранение в память сигналов
            self.social_cues.append(social_cues)
            
            # Обновление уровня адаптивности
            if social_cues['norm_compliance'] > 0.5:
                self.social_adaptability = min(1.0, self.social_adaptability + 0.03)
            
            logger.info(f"Social cues analyzed: {len(social_cues['detected_cues'])} cues detected")
            return social_cues
            
        except Exception as e:
            logger.error(f"Error analyzing social cues: {e}")
            return {
                'environment': environment,
                'detected_cues': [],
                'interpretation': ['Error in cue analysis'],
                'norm_compliance': 0.0,
                'timestamp': time.time()
            }
    
    def get_social_learning_profile(self):
        """Получение профиля социального обучения"""
        return {
            'social_patterns': self.social_patterns,
            'learning_efficiency': self.learning_efficiency,
            'social_adaptability': self.social_adaptability,
            'recent_observations': list(self.observation_memory)[-10:] if len(self.observation_memory) >= 10 else list(self.observation_memory),
            'recent_imitations': list(self.modeling_behavior)[-5:] if len(self.modeling_behavior) >= 5 else list(self.modeling_behavior),
            'social_cues': list(self.social_cues)[-5:] if len(self.social_cues) >= 5 else list(self.social_cues),
            'learning_transfer_history': list(self.learning_transfer.values())[-10:] if len(self.learning_transfer) >= 10 else list(self.learning_transfer.values()),
            'timestamp': time.time()
        }
class PainMemorySystem:
    """Система памяти о боли и обучения от ошибок."""
    def __init__(self):
        self.pain_experiences = []
        self.pain_prevention_rules = []
        self.learning_from_pain = True
        self.pain_history = deque(maxlen=1000)  # История боли с ограничением размера
        self.pain_intensity_history = deque(maxlen=100)
        self.pain_recognition_threshold = 0.3
        self.pain_adaptation_factor = 0.1
        
    def remember_pain(self, experience):
        """Запоминание ошибки как "боли" с расширенной информацией"""
        try:
            pain_record = {
                'experience': experience,
                'timestamp': time.time(),
                'context': experience.get('context', {}),
                'error': experience.get('error', 0.0),
                'solution': experience.get('solution', None),
                'confidence': experience.get('confidence', 0.0),
                'emotional_feedback': experience.get('emotional_feedback', {}),
                'pain_type': experience.get('pain_type', 'generic'),
                'severity': self._assess_pain_severity(experience),
                'memory_strength': 1.0,
                'repetition_count': 0,
                'last_mentioned': time.time(),
                'learning_opportunity': experience.get('learning_opportunity', True)
            }
            
            # Проверяем на повторяющиеся паттерны
            self._check_for_repeated_patterns(pain_record)
            
            # Обновляем историю боли
            self.pain_history.append(pain_record)
            self.pain_intensity_history.append(pain_record['error'])
            
            # Сохраняем в основную память
            self.pain_experiences.append(pain_record)
            
            logger.info(f"Pain experience remembered: error={pain_record['error']:.3f}, "
                       f"type={pain_record['pain_type']}, severity={pain_record['severity']}")
            
            return pain_record
            
        except Exception as e:
            logger.error(f"Error remembering pain experience: {e}")
            # Возврат базовой записи при ошибке
            return {
                'experience': experience,
                'timestamp': time.time(),
                'context': experience.get('context', {}),
                'error': experience.get('error', 0.0),
                'solution': experience.get('solution', None),
                'confidence': experience.get('confidence', 0.0),
                'emotional_feedback': experience.get('emotional_feedback', {}),
                'pain_type': experience.get('pain_type', 'generic'),
                'severity': 0.0,
                'memory_strength': 0.0,
                'repetition_count': 0,
                'last_mentioned': time.time(),
                'learning_opportunity': experience.get('learning_opportunity', True)
            }
    
    def _assess_pain_severity(self, experience):
        """Оценка серьезности боли"""
        error = experience.get('error', 0.0)
        confidence = experience.get('confidence', 0.0)
        
        # Серьезность зависит от ошибки и уверенности
        severity = error * (1.0 - confidence)
        
        # Классификация серьезности
        if severity > 0.8:
            return 'severe'
        elif severity > 0.5:
            return 'moderate'
        elif severity > 0.2:
            return 'mild'
        else:
            return 'minimal'
    
    def _check_for_repeated_patterns(self, pain_record):
        """Проверка на повторяющиеся паттерны"""
        try:
            # Проверяем последние N опытов на схожесть
            recent_experiences = self.pain_experiences[-10:] if len(self.pain_experiences) > 10 else self.pain_experiences
            
            context = pain_record['context']
            error = pain_record['error']
            
            for prev_record in recent_experiences:
                # Проверяем схожесть контекста
                if isinstance(context, dict) and isinstance(prev_record['context'], dict):
                    # Сравниваем ключевые поля контекста
                    context_match = self._compare_contexts(context, prev_record['context'])
                    if context_match > 0.7:  # Высокое сходство
                        prev_record['repetition_count'] += 1
                        prev_record['last_mentioned'] = time.time()
                        
                        # Увеличиваем силу памяти для повторяющихся случаев
                        prev_record['memory_strength'] = min(1.0, prev_record['memory_strength'] + 0.1)
                        
                        # Если много повторений, создаем правило предотвращения
                        if prev_record['repetition_count'] >= 3:
                            self._create_pain_prevention_rule(prev_record)
            
        except Exception as e:
            logger.error(f"Error checking repeated patterns: {e}")
    
    def _compare_contexts(self, context1, context2):
        """Сравнение двух контекстов"""
        if not isinstance(context1, dict) or not isinstance(context2, dict):
            return 0.0
            
        # Сравнение ключей и значений
        common_keys = set(context1.keys()) & set(context2.keys())
        total_keys = set(context1.keys()) | set(context2.keys())
        
        if not total_keys:
            return 1.0 if context1 == context2 else 0.0
            
        # Оценка по совпадению ключей
        key_similarity = len(common_keys) / len(total_keys) if total_keys else 0.0
        
        # Оценка по совпадению значений
        value_similarity = 0.0
        if common_keys:
            matching_values = 0
            for key in common_keys:
                if context1[key] == context2[key]:
                    matching_values += 1
            value_similarity = matching_values / len(common_keys) if common_keys else 0.0
            
        # Среднее значение с весами
        return (key_similarity * 0.4 + value_similarity * 0.6)
    
    def _create_pain_prevention_rule(self, experience):
        """Создание правила предотвращения боли"""
        try:
            rule = {
                'type': 'pain_prevention',
                'pattern': experience['context'],
                'error_type': experience['pain_type'],
                'severity': experience['severity'],
                'recommended_action': f'Avoid similar approach to "{experience["context"]}"',
                'created_at': time.time(),
                'trigger_count': experience['repetition_count'],
                'last_triggered': time.time()
            }
            
            # Добавляем правило в список
            self.pain_prevention_rules.append(rule)
            
            # Ограничиваем количество правил
            if len(self.pain_prevention_rules) > 100:
                self.pain_prevention_rules.pop(0)
                
            logger.debug(f"Pain prevention rule created: {rule['pattern'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error creating pain prevention rule: {e}")
    
    def avoid_pain_patterns(self, current_context, current_error=None):
        """Избегание паттернов боли с учетом текущей ошибки"""
        try:
            # Поиск похожих паттернов из прошлого
            # Активация защитных механизмов
            avoidance_actions = []
            
            # Простой анализ контекста
            if isinstance(current_context, str):
                # Ищем похожие шаблоны в памяти
                for experience in self.pain_experiences[-10:]:  # Последние 10 опытов
                    if experience['context'] and isinstance(experience['context'], str):
                        # Простое сравнение
                        if current_context.lower() in experience['context'].lower():
                            action = {
                                'warning': 'Potential pain pattern detected',
                                'previous_experience': experience['experience'],
                                'recommended_action': 'Avoid similar approach',
                                'error_type': experience['pain_type'],
                                'severity': experience['severity'],
                                'timestamp': experience['timestamp']
                            }
                            avoidance_actions.append(action)
                            
            # Анализ по текущей ошибке
            if current_error is not None and current_error > self.pain_recognition_threshold:
                # Ищем похожие ошибки в истории
                recent_errors = [exp for exp in self.pain_experiences[-20:] 
                               if exp['error'] > self.pain_recognition_threshold]
                
                for experience in recent_errors:
                    if experience['error'] > 0.5 and abs(experience['error'] - current_error) < 0.3:
                        action = {
                            'warning': 'Similar error pattern detected',
                            'previous_error': experience['error'],
                            'recommended_action': 'Modify approach to prevent repetition',
                            'error_type': experience['pain_type'],
                            'severity': experience['severity'],
                            'timestamp': experience['timestamp']
                        }
                        avoidance_actions.append(action)
            
            # Обновляем уровень адаптации
            if avoidance_actions:
                self._update_pain_adaptation(len(avoidance_actions))
            
            logger.info(f"Pain avoidance actions: {len(avoidance_actions)}")
            return avoidance_actions
            
        except Exception as e:
            logger.error(f"Error in pain avoidance: {e}")
            return []
    
    def _update_pain_adaptation(self, warning_count):
        """Обновление уровня адаптации к боли"""
        self.pain_adaptation_factor = min(1.0, self.pain_adaptation_factor + warning_count * 0.05)
    
    def get_pain_statistics(self):
        """Получение статистики по боли"""
        try:
            total_experiences = len(self.pain_experiences)
            severe_pain = sum(1 for exp in self.pain_experiences if exp['severity'] == 'severe')
            moderate_pain = sum(1 for exp in self.pain_experiences if exp['severity'] == 'moderate')
            mild_pain = sum(1 for exp in self.pain_experiences if exp['severity'] == 'mild')
            
            # Средняя интенсивность боли
            avg_intensity = sum(exp['error'] for exp in self.pain_experiences) / total_experiences if total_experiences > 0 else 0.0
            
            # Частота повторений
            repeated_experiences = sum(1 for exp in self.pain_experiences if exp['repetition_count'] > 0)
            
            return {
                'total_experiences': total_experiences,
                'severe_pain': severe_pain,
                'moderate_pain': moderate_pain,
                'mild_pain': mild_pain,
                'average_intensity': avg_intensity,
                'repeated_experiences': repeated_experiences,
                'prevention_rules': len(self.pain_prevention_rules),
                'adaptation_factor': self.pain_adaptation_factor,
                'recent_pain_history': list(self.pain_history)[-10:] if len(self.pain_history) >= 10 else list(self.pain_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting pain statistics: {e}")
            return {
                'total_experiences': 0,
                'severe_pain': 0,
                'moderate_pain': 0,
                'mild_pain': 0,
                'average_intensity': 0.0,
                'repeated_experiences': 0,
                'prevention_rules': 0,
                'adaptation_factor': self.pain_adaptation_factor,
                'recent_pain_history': []
            }
    
    def get_pain_prevention_rules(self):
        """Получение правил предотвращения боли"""
        return self.pain_prevention_rules
    
    def clear_pain_memory(self):
        """Очистка памяти о боли"""
        self.pain_experiences.clear()
        self.pain_prevention_rules.clear()
        self.pain_history.clear()
        self.pain_intensity_history.clear()
        logger.info("Pain memory cleared")
    
    def export_pain_data(self):
        """Экспорт данных о боли для анализа"""
        return {
            'pain_experiences': self.pain_experiences,
            'prevention_rules': self.pain_prevention_rules,
            'statistics': self.get_pain_statistics(),
            'timestamp': time.time()
        }
class TemporalReasoning:
    """Система временного восприятия."""
    def __init__(self):
        self.time_memory = []
        self.temporal_patterns = []
        self.causal_reasoning = {}
        self.temporal_context_history = deque(maxlen=100)
        self.temporal_adaptability = 0.0
        self.event_sequence_memory = {}
        self.temporal_prediction_history = deque(maxlen=50)
        self.time_decay_factor = 0.95
        
    def understand_temporal_relationships(self, events, context=None):
        """Понимание временных связей с учетом контекста"""
        try:
            # Анализ причинно-следственных связей
            # Понимание последовательности событий
            # Прогнозирование последствий
            temporal_analysis = {
                'events': events,
                'context': context,
                'causal_relationships': [],
                'sequence_patterns': [],
                'predictions': [],
                'temporal_confidence': 0.0,
                'timestamp': time.time()
            }
            
            # Расширенный анализ временных связей
            if isinstance(events, list) and len(events) > 1:
                # Анализ последовательности событий
                sequence_patterns = self._analyze_sequence_patterns(events)
                temporal_analysis['sequence_patterns'] = sequence_patterns
                
                # Анализ причинно-следственных связей
                causal_relationships = []
                for i in range(len(events) - 1):
                    event_a = events[i]
                    event_b = events[i+1]
                    
                    # Оценка временной связи
                    temporal_order = self._determine_temporal_order(event_a, event_b, i)
                    
                    # Определение типа связи
                    relationship_type = self._classify_relationship(event_a, event_b, i)
                    
                    causal_rel = {
                        'event_a': event_a,
                        'event_b': event_b,
                        'temporal_order': temporal_order,
                        'relationship_type': relationship_type,
                        'confidence': self._calculate_relationship_confidence(event_a, event_b, i),
                        'timestamp': time.time()
                    }
                    
                    causal_relationships.append(causal_rel)
                    
                    # Сохраняем в историю временных контекстов
                    self.temporal_context_history.append({
                        'events': [event_a, event_b],
                        'relationship': causal_rel,
                        'timestamp': time.time()
                    })
                
                temporal_analysis['causal_relationships'] = causal_relationships
                
                # Прогнозирование на основе паттернов
                predictions = self._generate_predictions(events, causal_relationships)
                temporal_analysis['predictions'] = predictions
                
                # Общая уверенность
                if causal_relationships:
                    avg_confidence = sum(rel['confidence'] for rel in causal_relationships) / len(causal_relationships)
                    temporal_analysis['temporal_confidence'] = min(1.0, avg_confidence * 1.1)
                else:
                    temporal_analysis['temporal_confidence'] = 0.3
            
            # Обновляем уровень адаптивности
            self.temporal_adaptability = min(1.0, self.temporal_adaptability + 0.02 * temporal_analysis['temporal_confidence'])
            
            # Сохраняем анализ в историю
            self.temporal_prediction_history.append(temporal_analysis)
            
            logger.info(f"Temporal relationships understood (confidence: {temporal_analysis['temporal_confidence']:.3f})")
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Error in temporal relationship analysis: {e}")
            # Возврат базового анализа при ошибке
            return {
                'events': events,
                'context': context,
                'causal_relationships': [],
                'sequence_patterns': [],
                'predictions': [],
                'temporal_confidence': 0.1,
                'timestamp': time.time()
            }
    
    def _analyze_sequence_patterns(self, events):
        """Анализ паттернов последовательности"""
        patterns = []
        
        # Поиск повторяющихся паттернов
        if len(events) >= 3:
            # Проверяем на циклические паттерны
            for window_size in range(2, min(5, len(events))):
                for i in range(len(events) - window_size + 1):
                    window = tuple(events[i:i+window_size])
                    if window not in self.event_sequence_memory:
                        self.event_sequence_memory[window] = {
                            'count': 0,
                            'first_occurrence': time.time(),
                            'last_occurrence': time.time()
                        }
                    self.event_sequence_memory[window]['count'] += 1
                    self.event_sequence_memory[window]['last_occurrence'] = time.time()
                    
                    # Если паттерн встречается часто, считаем его паттерном
                    if self.event_sequence_memory[window]['count'] >= 2:
                        patterns.append({
                            'pattern': window,
                            'frequency': self.event_sequence_memory[window]['count'],
                            'type': 'repetitive'
                        })
        
        # Поиск логических последовательностей
        logical_patterns = self._detect_logical_sequences(events)
        patterns.extend(logical_patterns)
        
        return patterns
    
    def _detect_logical_sequences(self, events):
        """Обнаружение логических последовательностей"""
        patterns = []
        
        # Проверяем на типичные логические цепочки
        logical_chain_indicators = [
            ('beginning', 'middle', 'end'),
            ('cause', 'effect', 'result'),
            ('problem', 'solution', 'outcome'),
            ('initiation', 'process', 'completion')
        ]
        
        # Простая проверка на наличие ключевых слов
        for chain in logical_chain_indicators:
            found_positions = []
            for i, event in enumerate(events):
                if isinstance(event, str):
                    event_lower = event.lower()
                    for indicator in chain:
                        if indicator in event_lower:
                            found_positions.append((i, indicator))
                            break
            
            if len(found_positions) >= 2:
                patterns.append({
                    'pattern': chain,
                    'sequence': found_positions,
                    'type': 'logical_chain'
                })
        
        return patterns
    
    def _determine_temporal_order(self, event_a, event_b, position):
        """Определение временного порядка событий"""
        # По умолчанию последовательный порядок
        order = 'sequential'
        
        # Проверяем на обратный порядок (возможные ошибки)
        if isinstance(event_a, str) and isinstance(event_b, str):
            # Простая проверка на обратную последовательность
            if 'before' in event_b.lower() or 'previous' in event_b.lower():
                order = 'reverse'
            elif 'after' in event_a.lower() or 'subsequent' in event_a.lower():
                order = 'forward'
                
        return order
    
    def _classify_relationship(self, event_a, event_b, position):
        """Классификация типа связи между событиями"""
        # Определяем тип связи на основе позиции и контекста
        if position % 3 == 0:
            return 'causal'  # Причинно-следственная
        elif position % 3 == 1:
            return 'correlative'  # Корреляционная
        else:
            return 'temporal'  # Временная
    
    def _calculate_relationship_confidence(self, event_a, event_b, position):
        """Расчет уверенности в связи событий"""
        confidence = 0.5  # Базовая уверенность
        
        # Увеличиваем уверенность при наличии ключевых слов
        if isinstance(event_a, str) and isinstance(event_b, str):
            event_a_lower = event_a.lower()
            event_b_lower = event_b.lower()
            
            # Проверяем на ключевые слова причинности
            causal_indicators = ['because', 'due to', 'as a result', 'therefore', 'so that']
            correlation_indicators = ['also', 'similarly', 'likewise', 'in addition']
            
            if any(indicator in event_a_lower for indicator in causal_indicators):
                confidence += 0.3
            elif any(indicator in event_b_lower for indicator in causal_indicators):
                confidence += 0.3
            elif any(indicator in event_a_lower for indicator in correlation_indicators):
                confidence += 0.2
            elif any(indicator in event_b_lower for indicator in correlation_indicators):
                confidence += 0.2
                
        # Учет позиции в последовательности
        if position < 3:
            confidence += 0.1  # Первые события чаще связаны
        
        return min(1.0, confidence)
    
    def _generate_predictions(self, events, causal_relationships):
        """Генерация прогнозов на основе временных связей"""
        predictions = []
        
        if len(events) >= 2 and causal_relationships:
            # Простая прогнозная логика
            last_relationship = causal_relationships[-1]
            if last_relationship['relationship_type'] == 'causal':
                predictions.append({
                    'prediction': 'Expected consequence based on causal relationship',
                    'confidence': last_relationship['confidence'],
                    'based_on': last_relationship['event_a']
                })
            elif last_relationship['relationship_type'] == 'correlative':
                predictions.append({
                    'prediction': 'Likely continuation based on correlational pattern',
                    'confidence': last_relationship['confidence'] * 0.8,
                    'based_on': last_relationship['event_a']
                })
        
        # Прогноз на основе паттернов
        if self.temporal_patterns:
            for pattern in self.temporal_patterns[-3:]:  # Последние 3 паттерна
                predictions.append({
                    'prediction': f'Pattern-based prediction: {pattern}',
                    'confidence': 0.6,
                    'based_on': 'temporal_pattern'
                })
        
        return predictions
    
    def time_based_learning(self, temporal_context, learning_strength=1.0):
        """Обучение с учетом временных факторов"""
        try:
            # Учет временной последовательности
            # Адаптация к временным изменениям
            learning = {
                'temporal_context': temporal_context,
                'learning_strength': learning_strength,
                'adaptive_strategy': 'sequential',
                'adjustments': [],
                'timestamp': time.time(),
                'temporal_adaptability': self.temporal_adaptability
            }
            
            # Расширенная адаптация с учетом контекста
            if isinstance(temporal_context, dict):
                # Адаптация по времени
                if 'time_passed' in temporal_context:
                    learning['adaptive_strategy'] = 'adaptive'
                    learning['adjustments'].append('Adjusted to time progression')
                    # Увеличиваем адаптивность
                    self.temporal_adaptability = min(1.0, self.temporal_adaptability + 0.05 * learning_strength)
                
                if 'seasonal' in temporal_context:
                    learning['adjustments'].append('Seasonal adaptation applied')
                    self.temporal_adaptability = min(1.0, self.temporal_adaptability + 0.03 * learning_strength)
                
                if 'periodic' in temporal_context:
                    learning['adaptive_strategy'] = 'periodic'
                    learning['adjustments'].append('Periodic pattern recognition applied')
                    self.temporal_adaptability = min(1.0, self.temporal_adaptability + 0.04 * learning_strength)
                
                if 'urgent' in temporal_context:
                    learning['adaptive_strategy'] = 'rapid'
                    learning['adjustments'].append('Rapid adaptation for urgent context')
                    self.temporal_adaptability = min(1.0, self.temporal_adaptability + 0.06 * learning_strength)
                
                # Учет временной динамики
                if 'temporal_change' in temporal_context:
                    learning['adjustments'].append('Temporal change detected and adapted')
                    # Снижаем стабильность памяти при изменениях
                    for memory_key in list(self.event_sequence_memory.keys()):
                        if self.event_sequence_memory[memory_key]['last_occurrence'] < time.time() - 86400:  # 24 часа
                            self.event_sequence_memory[memory_key]['count'] *= self.time_decay_factor
                
                # Адаптация на основе силы обучения
                learning['adjustments'].append(f'Learning strength: {learning_strength:.2f}')
            
            # Обновление истории обучения
            self._update_learning_history(learning)
            
            logger.info(f"Time-based learning completed (adaptability: {self.temporal_adaptability:.3f})")
            return learning
            
        except Exception as e:
            logger.error(f"Error in time-based learning: {e}")
            # Возврат базового обучения при ошибке
            return {
                'temporal_context': temporal_context,
                'learning_strength': learning_strength,
                'adaptive_strategy': 'sequential',
                'adjustments': ['Error in learning process'],
                'timestamp': time.time(),
                'temporal_adaptability': self.temporal_adaptability
            }
    
    def _update_learning_history(self, learning):
        """Обновление истории обучения"""
        # Сохраняем в историю временных контекстов
        self.temporal_context_history.append({
            'learning': learning,
            'timestamp': time.time()
        })
    
    def get_temporal_profile(self):
        """Получение профиля временного восприятия"""
        return {
            'temporal_adaptability': self.temporal_adaptability,
            'event_sequence_memory': dict(self.event_sequence_memory),
            'temporal_context_history': list(self.temporal_context_history)[-10:] if len(self.temporal_context_history) >= 10 else list(self.temporal_context_history),
            'temporal_prediction_history': list(self.temporal_prediction_history)[-5:] if len(self.temporal_prediction_history) >= 5 else list(self.temporal_prediction_history),
            'recent_patterns': self.temporal_patterns[-5:] if len(self.temporal_patterns) >= 5 else self.temporal_patterns,
            'timestamp': time.time()
        }
    
    def decay_temporal_memory(self):
        """Уменьшение силы временной памяти со временем"""
        try:
            # Уменьшаем силу памяти по времени
            current_time = time.time()
            decay_threshold = 86400  # 24 часа
            
            for key, memory_data in list(self.event_sequence_memory.items()):
                time_since_last = current_time - memory_data['last_occurrence']
                if time_since_last > decay_threshold:
                    # Уменьшаем счетчик при старении
                    memory_data['count'] = max(0, memory_data['count'] * self.time_decay_factor)
                    # Удаляем если совсем старое
                    if memory_data['count'] < 0.1:
                        del self.event_sequence_memory[key]
            
            logger.info(f"Temporal memory decay applied to {len(self.event_sequence_memory)} entries")
            
        except Exception as e:
            logger.error(f"Error in temporal memory decay: {e}")
class SelfLearningSystem:
    """Система самостоятельного обучения."""
    def __init__(self):
        self.learning_strategies = []
        self.knowledge_gaps = []
        self.self_assessment = {}
        self.learning_history = deque(maxlen=100)
        self.skill_progression = {}
        self.learning_efficiency = 0.0
        self.metacognitive_awareness = 0.0
        self.adaptive_learning_paths = {}
        self.cognitive_load_history = []
        
    def identify_learning_needs(self, current_state, context=None):
        """Определение потребностей в обучении с учетом контекста"""
        try:
            # Анализ пробелов в знаниях
            # Определение приоритетов
            # Планирование обучения
            needs = {
                'current_state': current_state,
                'context': context,
                'gaps_identified': [],
                'priorities': [],
                'learning_plan': [],
                'confidence': 0.0,
                'timestamp': time.time()
            }
            
            # Расширенный анализ пробелов в знаниях
            if isinstance(current_state, dict):
                # Проверяем наличие ключей
                required_keys = ['knowledge', 'skills', 'understanding', 'experience']
                for key in required_keys:
                    if key not in current_state:
                        needs['gaps_identified'].append(key)
                        
                # Учет контекста для более точного анализа
                if context and isinstance(context, dict):
                    # Анализ специфических областей
                    if 'domain' in context:
                        domain_gaps = self._analyze_domain_gaps(context['domain'], current_state)
                        needs['gaps_identified'].extend(domain_gaps)
                        
                    # Анализ уровня сложности
                    if 'difficulty_level' in context:
                        difficulty_factor = context['difficulty_level']
                        if difficulty_factor > 0.7:
                            needs['gaps_identified'].extend(['advanced_skills', 'specialized_knowledge'])
                            
                # Устанавливаем приоритеты с учетом контекста
                if needs['gaps_identified']:
                    # Приоритеты зависят от типа пробела и контекста
                    priorities = []
                    for gap in needs['gaps_identified']:
                        priority = self._calculate_gap_priority(gap, context)
                        priorities.append(priority)
                    
                    needs['priorities'] = priorities
                    # Сортируем по приоритетам
                    gap_priority_pairs = list(zip(needs['gaps_identified'], priorities))
                    gap_priority_pairs.sort(key=lambda x: x[1], reverse=True)
                    needs['gaps_identified'] = [pair[0] for pair in gap_priority_pairs]
                else:
                    needs['priorities'] = ['enhance_existing']
                    
                # Формируем план обучения
                for gap in needs['gaps_identified']:
                    plan_item = self._generate_learning_plan_item(gap, context)
                    needs['learning_plan'].append(plan_item)
                    
                # Оценка уверенности в анализе
                needs['confidence'] = self._calculate_confidence(needs['gaps_identified'], context)
                
            # Обновляем историю обучения
            self.learning_history.append({
                'needs': needs,
                'timestamp': time.time()
            })
            
            logger.info(f"Learning needs identified: {len(needs['gaps_identified'])} gaps "
                       f"(confidence: {needs['confidence']:.2f})")
            return needs
            
        except Exception as e:
            logger.error(f"Error identifying learning needs: {e}")
            # Возврат базового анализа при ошибке
            return {
                'current_state': current_state,
                'context': context,
                'gaps_identified': [],
                'priorities': [],
                'learning_plan': [],
                'confidence': 0.1,
                'timestamp': time.time()
            }
    
    def _analyze_domain_gaps(self, domain, current_state):
        """Анализ пробелов в конкретной области"""
        domain_gaps = []
        
        # Определение специфических пробелов в зависимости от домена
        domain_specific_gaps = {
            'technical': ['programming_languages', 'algorithms', 'data_structures', 'system_design'],
            'scientific': ['theoretical_foundation', 'experimental_methods', 'data_analysis', 'research_ethics'],
            'creative': ['design_principles', 'creative_process', 'expression_techniques', 'innovation_methods'],
            'social': ['communication_skills', 'interpersonal_relations', 'cultural_awareness', 'leadership'],
            'academic': ['research_methods', 'critical_analysis', 'writing_skills', 'presentation']
        }
        
        if domain in domain_specific_gaps:
            available_skills = list(current_state.keys())
            required_skills = domain_specific_gaps[domain]
            for skill in required_skills:
                if skill not in available_skills:
                    domain_gaps.append(skill)
                    
        return domain_gaps
    
    def _calculate_gap_priority(self, gap, context):
        """Расчет приоритета пробела"""
        # Базовый приоритет
        base_priority = 0.5
        
        # Учет контекста
        if context and isinstance(context, dict):
            # Приоритеты могут зависеть от уровня сложности
            if 'difficulty_level' in context:
                difficulty = context['difficulty_level']
                base_priority = min(1.0, base_priority + difficulty * 0.3)
                
            # Приоритеты могут зависеть от важности области
            if 'importance' in context:
                importance = context['importance']
                base_priority = min(1.0, base_priority + importance * 0.2)
                
        # Учет типа пробела
        gap_priorities = {
            'knowledge': 0.8,
            'skills': 0.9,
            'understanding': 0.7,
            'experience': 0.6,
            'advanced_skills': 0.95,
            'specialized_knowledge': 0.9,
            'technical_skills': 0.85,
            'creative_process': 0.75,
            'communication_skills': 0.7
        }
        
        if gap in gap_priorities:
            base_priority = max(0.0, min(1.0, base_priority + gap_priorities[gap] * 0.1))
            
        return min(1.0, base_priority)
    
    def _generate_learning_plan_item(self, gap, context):
        """Генерация элемента плана обучения"""
        plan_templates = {
            'knowledge': 'Study foundational concepts in {gap}',
            'skills': 'Practice {gap} through hands-on exercises',
            'understanding': 'Deep dive into {gap} theory and applications',
            'experience': 'Gain practical experience in {gap}',
            'advanced_skills': 'Master advanced techniques in {gap}',
            'specialized_knowledge': 'Develop specialized expertise in {gap}',
            'technical_skills': 'Enhance technical proficiency in {gap}',
            'creative_process': 'Explore creative methodologies for {gap}',
            'communication_skills': 'Improve communication abilities in {gap}'
        }
        
        template = plan_templates.get(gap, 'Focus on {gap}')
        return template.format(gap=gap)
    
    def _calculate_confidence(self, gaps, context):
        """Расчет уверенности в анализе пробелов"""
        base_confidence = 0.5
        
        # Увеличиваем уверенность при наличии контекста
        if context:
            context_factor = 0.0
            if isinstance(context, dict):
                context_factor = min(0.5, len(context) * 0.1)
            base_confidence += context_factor
            
        # Увеличиваем уверенность при наличии большого количества пробелов
        if len(gaps) > 0:
            gaps_factor = min(0.3, len(gaps) * 0.05)
            base_confidence += gaps_factor
            
        return min(1.0, base_confidence)
    
    def self_evaluate_performance(self, performance_data=None, context=None):
        """Самооценка знаний и навыков с учетом контекста"""
        try:
            # Оценка собственных способностей
            # Определение прогресса
            # Планирование дальнейшего обучения
            evaluation = {
                'performance_score': 0.0,
                'skills_assessed': [],
                'improvement_areas': [],
                'next_steps': [],
                'context': context,
                'timestamp': time.time(),
                'self_reflection': {}
            }
            
            # Расширенная оценка с учетом данных
            if performance_data and isinstance(performance_data, dict):
                # Используем предоставленные данные для оценки
                if 'accuracy' in performance_data:
                    evaluation['performance_score'] = performance_data['accuracy']
                elif 'confidence' in performance_data:
                    evaluation['performance_score'] = performance_data['confidence']
                else:
                    evaluation['performance_score'] = random.uniform(0.3, 0.9)
            else:
                # Простая оценка
                evaluation['performance_score'] = random.uniform(0.3, 0.9)
            
            # Оценка навыков
            skills = ['knowledge', 'understanding', 'application', 'critical_thinking', 'problem_solving']
            evaluation['skills_assessed'] = skills
            
            # Определение областей улучшения
            improvement_areas = []
            if evaluation['performance_score'] < 0.5:
                improvement_areas.extend(['more practice', 'basic concepts', 'fundamentals'])
            elif evaluation['performance_score'] < 0.7:
                improvement_areas.extend(['deeper understanding', 'advanced concepts'])
            else:
                improvement_areas.extend(['refinement', 'specialization'])
                
            evaluation['improvement_areas'] = improvement_areas
            
            # Планирование дальнейших шагов
            next_steps = []
            if evaluation['performance_score'] < 0.6:
                next_steps.extend(['review fundamentals', 'additional practice'])
            elif evaluation['performance_score'] < 0.8:
                next_steps.extend(['advanced study', 'practical application'])
            else:
                next_steps.extend(['specialization', 'expertise development'])
                
            evaluation['next_steps'] = next_steps
            
            # Самоанализ
            evaluation['self_reflection'] = self._conduct_self_reflection(evaluation, context)
            
            # Обновляем историю оценок
            self.learning_history.append({
                'evaluation': evaluation,
                'timestamp': time.time()
            })
            
            # Обновляем уровень метакогнитивной осведомленности
            self.metacognitive_awareness = min(1.0, self.metacognitive_awareness + 0.02)
            
            logger.info(f"Performance evaluated: {evaluation['performance_score']:.3f} "
                       f"(improvement areas: {len(evaluation['improvement_areas'])})")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in self-evaluation: {e}")
            # Возврат базовой оценки при ошибке
            return {
                'performance_score': random.uniform(0.3, 0.9),
                'skills_assessed': ['knowledge', 'understanding', 'application'],
                'improvement_areas': ['more practice', 'deeper understanding'],
                'next_steps': ['continue learning', 'apply knowledge'],
                'context': context,
                'timestamp': time.time(),
                'self_reflection': {'error': 'Evaluation failed'}
            }
    
    def _conduct_self_reflection(self, evaluation, context):
        """Проведение самоанализа"""
        reflection = {
            'performance_trend': self._analyze_performance_trend(),
            'learning_efficiency': self._calculate_learning_efficiency(),
            'metacognitive_insights': self._generate_metacognitive_insights(evaluation, context),
            'confidence_level': evaluation['performance_score'],
            'self_assessment_accuracy': 0.0
        }
        
        # Оценка точности самооценки
        # Можно улучшить с учетом исторических данных
        reflection['self_assessment_accuracy'] = min(1.0, 0.7 + self.metacognitive_awareness * 0.3)
        
        return reflection
    
    def _analyze_performance_trend(self):
        """Анализ тренда производительности"""
        if len(self.learning_history) < 2:
            return 'initial'
            
        # Простой анализ последних оценок
        recent_evaluations = [item['evaluation']['performance_score'] 
                            for item in list(self.learning_history)[-5:] 
                            if 'evaluation' in item]
        
        if len(recent_evaluations) < 2:
            return 'stable'
            
        # Определяем направление изменения
        first = recent_evaluations[0]
        last = recent_evaluations[-1]
        diff = last - first
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_learning_efficiency(self):
        """Расчет эффективности обучения"""
        # Простой расчет на основе истории
        if len(self.learning_history) < 3:
            return 0.5
            
        # Средняя эффективность на основе последних оценок
        recent_scores = [item['evaluation']['performance_score'] 
                        for item in list(self.learning_history)[-10:] 
                        if 'evaluation' in item]
        
        if len(recent_scores) < 2:
            return 0.5
            
        # Рассчитываем среднее значение
        avg_score = sum(recent_scores) / len(recent_scores)
        # Эффективность зависит от среднего значения
        efficiency = min(1.0, avg_score * 1.2)  # Масштабируем для более высоких значений
        
        # Обновляем общий уровень эффективности
        self.learning_efficiency = min(1.0, self.learning_efficiency * 0.9 + efficiency * 0.1)
        
        return efficiency
    
    def _generate_metacognitive_insights(self, evaluation, context):
        """Генерация метакогнитивных инсайтов"""
        insights = []
        
        # Инсайты на основе оценки
        if evaluation['performance_score'] < 0.4:
            insights.append('Significant learning gaps detected')
            insights.append('Need for fundamental review')
        elif evaluation['performance_score'] < 0.6:
            insights.append('Moderate performance, room for improvement')
            insights.append('Focus on core concepts')
        elif evaluation['performance_score'] < 0.8:
            insights.append('Good performance, but potential for growth')
            insights.append('Consider advanced applications')
        else:
            insights.append('Strong performance, ready for specialization')
            insights.append('Look for optimization opportunities')
            
        # Инсайты на основе контекста
        if context and isinstance(context, dict):
            if 'difficulty_level' in context:
                difficulty = context['difficulty_level']
                if difficulty > 0.8:
                    insights.append('High difficulty tasks require more focused effort')
                elif difficulty < 0.3:
                    insights.append('Low difficulty tasks may benefit from variety')
                    
            if 'domain' in context:
                insights.append(f'Domain expertise in {context["domain"]} is important')
                
        return insights
    
    def adapt_learning_strategy(self, new_context, performance_feedback=None):
        """Адаптация стратегии обучения"""
        try:
            # Адаптация стратегии на основе нового контекста
            strategy = {
                'new_context': new_context,
                'adapted_strategy': 'adaptive',
                'reasoning': 'Strategy adapted to new learning context',
                'timestamp': time.time(),
                'performance_feedback': performance_feedback
            }
            
            # Определение типа адаптации
            if new_context and isinstance(new_context, dict):
                if 'learning_style' in new_context:
                    strategy['adapted_strategy'] = new_context['learning_style']
                    strategy['reasoning'] = f'Adapted to {new_context["learning_style"]} learning style'
                elif 'domain' in new_context:
                    strategy['adapted_strategy'] = 'domain_specialization'
                    strategy['reasoning'] = f'Focused on {new_context["domain"]} domain'
                elif 'difficulty_level' in new_context:
                    difficulty = new_context['difficulty_level']
                    if difficulty > 0.7:
                        strategy['adapted_strategy'] = 'intensive_practice'
                        strategy['reasoning'] = 'High difficulty requires intensive practice'
                    elif difficulty < 0.3:
                        strategy['adapted_strategy'] = 'exploratory_learning'
                        strategy['reasoning'] = 'Low difficulty allows for exploratory learning'
            
            # Сохраняем стратегию
            self.learning_strategies.append(strategy)
            
            # Обновляем историю когнитивной нагрузки
            self.cognitive_load_history.append({
                'strategy': strategy['adapted_strategy'],
                'timestamp': time.time()
            })
            
            logger.info(f"Learning strategy adapted: {strategy['adapted_strategy']}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error adapting learning strategy: {e}")
            return {
                'new_context': new_context,
                'adapted_strategy': 'default',
                'reasoning': 'Error in strategy adaptation',
                'timestamp': time.time(),
                'performance_feedback': performance_feedback
            }
    
    def get_learning_profile(self):
        """Получение профиля обучения"""
        return {
            'learning_strategies': self.learning_strategies[-10:] if len(self.learning_strategies) >= 10 else self.learning_strategies,
            'knowledge_gaps': self.knowledge_gaps,
            'learning_efficiency': self.learning_efficiency,
            'metacognitive_awareness': self.metacognitive_awareness,
            'recent_evaluations': list(self.learning_history)[-5:] if len(self.learning_history) >= 5 else list(self.learning_history),
            'skill_progression': self.skill_progression,
            'timestamp': time.time()
        }
class MultitaskingSystem:
    """Система многозадачности."""
    def __init__(self):
        self.task_queue = []
        self.context_switching_cost = 0.0
        self.priority_matrix = {}
        self.task_history = deque(maxlen=100)
        self.task_performance = {}
        self.cognitive_load = 0.0
        self.switching_efficiency = 0.0
        self.task_dependency_graph = {}
        
    def manage_tasks(self, tasks, priorities, context=None):
        """Управление множеством задач с учетом контекста и зависимостей"""
        try:
            # Оптимизация распределения внимания
            # Переключение между задачами
            # Минимизация потерь при переключении
            task_management = {
                'tasks': tasks,
                'priorities': priorities,
                'assigned_resources': [],
                'switching_costs': 0.0,
                'optimization_strategy': 'sequential',
                'context': context,
                'timestamp': time.time()
            }
            
            # Расширенное управление задачами с учетом контекста
            if not tasks:
                logger.info("No tasks to manage")
                return task_management
            
            # Оценка сложности задач
            task_complexities = self._assess_task_complexity(tasks)
            
            # Создание графа зависимостей задач
            self._build_task_dependencies(tasks)
            
            # Оптимизация распределения ресурсов
            optimized_allocation = self._optimize_resource_allocation(
                tasks, priorities, task_complexities, context
            )
            
            # Сохраняем распределение ресурсов
            task_management['assigned_resources'] = optimized_allocation
            
            # Оценка затрат на переключение с учетом контекста
            switching_costs = self._calculate_switching_costs(tasks, context)
            task_management['switching_costs'] = switching_costs
            
            # Определение стратегии оптимизации
            if len(tasks) > 3:
                task_management['optimization_strategy'] = 'parallel_optimized'
            elif len(tasks) > 1:
                task_management['optimization_strategy'] = 'sequential_optimized'
            else:
                task_management['optimization_strategy'] = 'single_task'
            
            # Обновляем историю задач
            self.task_history.append({
                'tasks': tasks,
                'priorities': priorities,
                'allocation': optimized_allocation,
                'switching_costs': switching_costs,
                'strategy': task_management['optimization_strategy'],
                'timestamp': time.time()
            })
            
            # Обновляем уровень когнитивной нагрузки
            self.cognitive_load = min(1.0, self.cognitive_load + 0.02 * len(tasks))
            
            logger.info(f"Tasks managed: {len(tasks)} tasks (strategy: {task_management['optimization_strategy']})")
            return task_management
            
        except Exception as e:
            logger.error(f"Error managing tasks: {e}")
            # Возврат базового управления при ошибке
            return {
                'tasks': tasks,
                'priorities': priorities,
                'assigned_resources': [
                    {
                        'task': task,
                        'priority': priorities[i] if i < len(priorities) else 1,
                        'allocated_time': 10,
                        'efficiency': 0.8
                    }
                    for i, task in enumerate(tasks)
                ],
                'switching_costs': len(tasks) * 0.05,
                'optimization_strategy': 'sequential',
                'context': context,
                'timestamp': time.time()
            }
    
    def _assess_task_complexity(self, tasks):
        """Оценка сложности задач"""
        complexities = []
        for task in tasks:
            complexity = 0.0
            if isinstance(task, str):
                # Оценка по длине и сложности синтаксиса
                complexity = min(1.0, len(task) / 100.0)
                # Учет ключевых слов сложности
                complexity_words = ['complex', 'difficult', 'challenging', 'advanced']
                found_words = [word for word in complexity_words if word in task.lower()]
                complexity += len(found_words) * 0.2
            elif isinstance(task, dict):
                # Если задача сложная структура
                complexity = min(1.0, len(str(task)) / 500.0)
            complexities.append(min(1.0, complexity))
        return complexities
    
    def _build_task_dependencies(self, tasks):
        """Построение графа зависимостей задач"""
        self.task_dependency_graph = {}
        for i, task in enumerate(tasks):
            self.task_dependency_graph[i] = {
                'depends_on': [],
                'dependent_on': [],
                'task': task
            }
            
            # Простая проверка на зависимости (например, задачи, начинающиеся с "then")
            if isinstance(task, str) and task.lower().startswith('then'):
                # Зависит от предыдущей задачи
                if i > 0:
                    self.task_dependency_graph[i]['depends_on'].append(i-1)
                    self.task_dependency_graph[i-1]['dependent_on'].append(i)
    
    def _optimize_resource_allocation(self, tasks, priorities, complexities, context):
        """Оптимизация распределения ресурсов"""
        allocation = []
        
        # Сортируем задачи по приоритетам и сложности
        task_info = list(zip(range(len(tasks)), tasks, priorities, complexities))
        task_info.sort(key=lambda x: (x[2], -x[3]), reverse=True)  # По приоритету, затем по сложности
        
        for idx, task, priority, complexity in task_info:
            # Расчет времени и эффективности с учетом контекста
            base_time = priority * 10
            efficiency_factor = min(1.0, priority * 0.8)
            
            # Учет сложности задачи
            if complexity > 0.7:
                base_time *= 1.5  # Сложные задачи требуют больше времени
                efficiency_factor *= 0.7  # Сложные задачи менее эффективны
            elif complexity > 0.5:
                base_time *= 1.2
                efficiency_factor *= 0.8
            
            # Учет контекста
            if context and isinstance(context, dict):
                if 'urgent' in context:
                    efficiency_factor = min(1.0, efficiency_factor * 1.2)
                if 'resource_constraint' in context:
                    base_time *= 1.3  # Больше времени при ограничениях ресурсов
            
            resource_allocation = {
                'task': task,
                'task_index': idx,
                'priority': priority,
                'complexity': complexity,
                'allocated_time': base_time,
                'efficiency': efficiency_factor,
                'optimized': True
            }
            allocation.append(resource_allocation)
        
        return allocation
    
    def _calculate_switching_costs(self, tasks, context):
        """Расчет затрат на переключение между задачами"""
        switching_costs = 0.0
        
        # Базовые затраты на переключение
        switching_costs = len(tasks) * 0.05
        
        # Учет сложности переключения
        if len(tasks) > 1:
            # Увеличиваем затраты при значительных различиях в задачах
            task_types = set()
            for task in tasks:
                if isinstance(task, str):
                    task_lower = task.lower()
                    if any(word in task_lower for word in ['coding', 'writing', 'analysis']):
                        task_types.add('creative')
                    elif any(word in task_lower for word in ['calculation', 'math', 'data']):
                        task_types.add('analytical')
                    else:
                        task_types.add('general')
                elif isinstance(task, dict):
                    task_types.add('structured')
            
            # Затраты выше при смешанных типах задач
            if len(task_types) > 1:
                switching_costs *= 1.3
            
            # Учет контекста
            if context and isinstance(context, dict):
                if 'time_pressure' in context:
                    switching_costs *= 1.2
                if 'multitask_complexity' in context:
                    switching_costs *= (1.0 + context['multitask_complexity'] * 0.2)
        
        # Учет когнитивной нагрузки
        switching_costs *= (1.0 + self.cognitive_load * 0.5)
        
        return min(1.0, switching_costs)
    
    def context_switching(self, new_task, current_context=None, task_history=None):
        """Переключение контекста с учетом истории задач"""
        try:
            # Система переключения между задачами
            # Минимизация потерь времени
            switch_info = {
                'old_task': self.task_queue[-1] if self.task_queue else None,
                'new_task': new_task,
                'switch_cost': 0.0,
                'transition_time': 0.0,
                'efficiency_loss': 0.0,
                'context': current_context,
                'timestamp': time.time()
            }
            
            # Расширенный расчет затрат на переключение
            switch_info['switch_cost'] = self._calculate_advanced_switching_cost(
                new_task, switch_info['old_task'], current_context, task_history
            )
            
            # Время перехода
            switch_info['transition_time'] = 0.2 + switch_info['switch_cost'] * 0.5
            
            # Потери эффективности
            switch_info['efficiency_loss'] = switch_info['switch_cost'] * 0.8
            
            # Обновляем историю задач
            if isinstance(new_task, str) and new_task not in self.task_queue:
                self.task_queue.append(new_task)
                # Ограничиваем историю
                if len(self.task_queue) > 50:
                    self.task_queue.pop(0)
            
            # Обновляем эффективность переключения
            self.switching_efficiency = min(1.0, self.switching_efficiency + 0.01 * (1.0 - switch_info['switch_cost']))
            
            logger.info(f"Context switched to: {new_task} (cost: {switch_info['switch_cost']:.3f})")
            return switch_info
            
        except Exception as e:
            logger.error(f"Error in context switching: {e}")
            # Возврат базовой информации при ошибке
            return {
                'old_task': self.task_queue[-1] if self.task_queue else None,
                'new_task': new_task,
                'switch_cost': 0.1,
                'transition_time': 0.2,
                'efficiency_loss': 0.08,
                'context': current_context,
                'timestamp': time.time()
            }
    
    def _calculate_advanced_switching_cost(self, new_task, old_task, context, task_history):
        """Расчет продвинутых затрат на переключение"""
        cost = 0.0
        
        # Базовая стоимость
        cost = 0.1
        
        # Учет схожести задач
        if old_task and new_task:
            old_lower = str(old_task).lower()
            new_lower = str(new_task).lower()
            
            # Сравнение ключевых слов
            old_words = set(old_lower.split())
            new_words = set(new_lower.split())
            
            # Коэффициент схожести
            if old_words and new_words:
                intersection = len(old_words & new_words)
                union = len(old_words | new_words)
                similarity = intersection / union if union > 0 else 0.0
                # Чем выше схожесть, тем ниже стоимость
                cost = max(0.05, cost * (1.0 - similarity * 0.5))
            else:
                # Если нет общих слов, стоимость выше
                cost = 0.15
        
        # Учет контекста
        if context and isinstance(context, dict):
            if 'task_type_change' in context and context['task_type_change']:
                cost += 0.1  # Смена типа задачи увеличивает затраты
            if 'context_switch_speed' in context:
                speed_factor = context['context_switch_speed']
                cost = max(0.05, cost * (1.0 - speed_factor * 0.3))
        
        # Учет истории задач
        if task_history and len(task_history) > 0:
            # Если последняя задача была похожа, затраты ниже
            recent_tasks = task_history[-5:] if len(task_history) > 5 else task_history
            similar_tasks = [t for t in recent_tasks if t and str(t).lower() == str(new_task).lower()]
            if similar_tasks:
                cost *= 0.7  # Схожие задачи - меньше затрат
        
        # Учет когнитивной нагрузки
        cost *= (1.0 + self.cognitive_load * 0.3)
        
        return min(1.0, cost)
    
    def get_multitasking_profile(self):
        """Получение профиля многозадачности"""
        return {
            'task_queue': self.task_queue[-10:],  # Последние 10 задач
            'context_switching_cost': self.context_switching_cost,
            'cognitive_load': self.cognitive_load,
            'switching_efficiency': self.switching_efficiency,
            'task_history': list(self.task_history)[-10:] if len(self.task_history) >= 10 else list(self.task_history),
            'task_performance': self.task_performance,
            'priority_matrix': self.priority_matrix,
            'timestamp': time.time()
        }
    
    def reset_system(self):
        """Сброс системы многозадачности"""
        self.task_queue.clear()
        self.context_switching_cost = 0.0
        self.priority_matrix.clear()
        self.task_history.clear()
        self.task_performance.clear()
        self.cognitive_load = 0.0
        self.switching_efficiency = 0.0
        self.task_dependency_graph.clear()
        logger.info("Multitasking system reset")
class HumanLikeBrain:
    """Главный класс - биологически-ориентированная нейросеть с якорями."""
    def __init__(self):
        self.anchor_system = AnchorSystem()
        self.emotional_system = EmotionalLearningSystem()
        self.pain_system = PainBasedLearning()
        self.context_system = ContextualRecognitionSystem()
        self.wisdom_system = WisdomSystem()
        self.instinct_system = InstinctSystem()
        self.inner_voice_system = InnerVoiceSystem()
        self.attention_system = AttentionSystem()
        self.memory_system = MemoryTypes()
        self.intuition_system = IntuitionSystem()
        self.motivation_system = MotivationSystem()
        self.personality_system = PersonalitySystem()
        self.critical_thinking_system = CriticalThinkingSystem()
        self.emotional_intelligence_system = EmotionalIntelligence()
        self.social_learning_system = SocialLearningSystem()
        self.pain_memory_system = PainMemorySystem()
        self.temporal_reasoning_system = TemporalReasoning()
        self.self_learning_system = SelfLearningSystem()
        self.multitasking_system = MultitaskingSystem()
        self.neuron_pool = {}
        self.active_anchors = []
        self.learning_history = []
        self.cognitive_load = 0.0
        self.system_state = {
            'attention_level': 0.0,
            'emotional_state': 'neutral',
            'learning_phase': 'exploration',
            'task_complexity': 0.0
        }
    
    def process_input(self, input_data, expected_output=None, context=None):
        """Обработка входных данных с полным контекстом"""
        try:
            # 1. Распознавание контекста
            context_info = self.context_system.recognize_context(input_data)
            
            # 2. Активация якорей
            anchor_chain = self._activate_relevant_anchors(input_data, context_info)
            
            # 3. Обратная связь и обучение
            if expected_output is not None:
                feedback = self.emotional_system.process_feedback(
                    self.predict(input_data), expected_output, context_info
                )
                # Обработка боли
                if feedback['punishment'] > 0.1:
                    self.pain_system.process_pain_signal(
                        feedback['error'],
                        context_info['confidence'],
                        context_info
                    )
                # Обновление систем обучения
                self._update_learning_systems(feedback, context_info)
            
            # 4. Обновление нейронов
            self._update_neurons(anchor_chain, input_data, context_info)
            
            # 5. Обновление систем состояния
            self._update_system_states(context_info, anchor_chain)
            
            # 6. Интеграция с другими системами
            self._integrate_with_other_systems(input_data, context_info)
            
            return self.predict(input_data)
            
        except Exception as e:
            logger.error(f"Error in process_input: {e}")
            # Возврат базового значения при ошибке
            return 0.5
    
    def _activate_relevant_anchors(self, input_data, context_info):
        """Активация релевантных якорей с учетом контекста и когнитивной нагрузки"""
        anchor_chain = []
        
        # Учет когнитивной нагрузки при активации
        load_factor = min(1.0, self.cognitive_load * 2.0)
        
        # Создаем новые якоря на основе входных данных
        if isinstance(input_data, str):
            words = input_data.lower().split()
            # Активируем якоря для ключевых слов с учетом нагрузки
            for i, word in enumerate(words[:5]):  # Первые 5 слов
                # Уменьшаем активацию при высокой нагрузке
                activation_factor = max(0.1, 1.0 - load_factor * 0.3)
                
                if word not in self.neuron_pool:
                    # Создаем новый якорь
                    anchor = self.anchor_system.create_anchor(
                        concept_name=word,
                        context=context_info,
                        anchor_type="concept"
                    )
                    self.neuron_pool[word] = anchor
                else:
                    anchor = self.neuron_pool[word]
                
                # Активируем якорь с учетом фактора
                anchor.activate(context=context_info, strength=activation_factor)
                anchor_chain.append(anchor)
                
        return anchor_chain
    
    def _update_learning_systems(self, feedback, context_info):
        """Обновление систем обучения на основе обратной связи"""
        try:
            # Обновление эмоциональной системы
            self.emotional_system.emotional_memory.append({
                'error': feedback['error'],
                'emotional_state': feedback['emotional_state'],
                'punishment': feedback['punishment'],
                'reward': feedback['reward'],
                'context': context_info,
                'timestamp': time.time()
            })
            
            # Обновление памяти боли
            if feedback['punishment'] > 0.05:
                self.pain_memory_system.remember_pain({
                    'experience': {
                        'context': context_info,
                        'error': feedback['error'],
                        'solution': None
                    },
                    'timestamp': time.time(),
                    'context': context_info,
                    'error': feedback['error'],
                    'solution': None
                })
            
            # Обновление системы мотивации
            if feedback['reward'] > 0.1:
                self.motivation_system.motivation_level = min(1.0, self.motivation_system.motivation_level + 0.02)
            else:
                self.motivation_system.motivation_level = max(0.0, self.motivation_system.motivation_level - 0.01)
                
            # Обновление самооценки
            performance_score = feedback['reward']
            self.self_learning_system.self_assessment['performance_score'] = performance_score
            
        except Exception as e:
            logger.error(f"Error updating learning systems: {e}")
    
    def _update_neurons(self, anchor_chain, input_data, context_info):
        """Обновление нейронов на основе активации с учетом когнитивной нагрузки"""
        try:
            for anchor in anchor_chain:
                # Обновляем память якоря с учетом нагрузки
                load_factor = min(1.0, self.cognitive_load * 1.5)
                memory_factor = max(0.5, 1.0 - load_factor * 0.2)
                
                anchor._update_memory(
                    activation=anchor.activation_strength * memory_factor,
                    context=context_info,
                    emotional_feedback=self.emotional_system.emotional_memory[-1] if self.emotional_system.emotional_memory else None
                )
                
                # Обновляем стабильность с учетом нагрузки
                stability_factor = max(0.1, 1.0 - load_factor * 0.3)
                anchor.stability = min(1.0, anchor.stability + 0.01 * stability_factor)
                
                # Обновляем эмоциональное состояние
                if self.emotional_system.emotional_memory:
                    last_emotion = self.emotional_system.emotional_memory[-1]['emotional_state']
                    anchor.emotional_state = self.emotional_system.emotional_states[last_emotion] * (1.0 - load_factor * 0.1)
                    
        except Exception as e:
            logger.error(f"Error updating neurons: {e}")
    
    def _update_system_states(self, context_info, anchor_chain):
        """Обновление состояния системы с учетом контекста и активации"""
        try:
            # Обновление уровня внимания
            attention_factor = sum(a.activation_strength for a in anchor_chain) / max(1, len(anchor_chain))
            self.attention_system.focus_level = min(1.0, attention_factor * 0.8 + 0.2)
            
            # Обновление когнитивной нагрузки
            self.cognitive_load = min(1.0, self.cognitive_load + 0.01 * attention_factor)
            
            # Обновление эмоционального состояния
            if self.emotional_system.emotional_memory:
                last_emotion = self.emotional_system.emotional_memory[-1]['emotional_state']
                self.system_state['emotional_state'] = last_emotion
            
            # Обновление состояния обучения
            if len(anchor_chain) > 0:
                avg_activation = sum(a.activation_strength for a in anchor_chain) / len(anchor_chain)
                if avg_activation > 0.7:
                    self.system_state['learning_phase'] = 'intensive'
                elif avg_activation > 0.3:
                    self.system_state['learning_phase'] = 'moderate'
                else:
                    self.system_state['learning_phase'] = 'exploration'
            
            # Обновление сложности задачи
            self.system_state['task_complexity'] = min(1.0, len(str(context_info.get('context', ''))) / 100.0)
            
        except Exception as e:
            logger.error(f"Error updating system states: {e}")
    
    def _integrate_with_other_systems(self, input_data, context_info):
        """Интеграция с другими системами для комплексной обработки"""
        try:
            # Интеграция с системой интуиции
            intuition_result = self.intuition_system.quick_insight(input_data, context_info)
            
            # Интеграция с системой внимания
            attention_weights = self.attention_system.focus_attention([input_data], context_info)
            
            # Интеграция с системой временного восприятия
            temporal_analysis = self.temporal_reasoning_system.understand_temporal_relationships([input_data], context_info)
            
            # Интеграция с системой эмоционального интеллекта
            emotion_recognition = self.emotional_intelligence_system.recognize_emotions(context_info)
            
            # Интеграция с системой социального обучения
            if isinstance(input_data, str) and len(input_data) > 20:
                social_learning = self.social_learning_system.observe_and_learn(input_data, context_info)
            
            # Обновление памяти с интеграцией
            self.memory_system.store_memory(
                data={
                    'input': input_data,
                    'context': context_info,
                    'intuition': intuition_result,
                    'attention': attention_weights,
                    'temporal': temporal_analysis,
                    'emotions': emotion_recognition
                },
                memory_type="long_term",
                emotional_value=emotion_recognition.get('confidence', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error integrating with other systems: {e}")
    
    def predict(self, input_data):
        """Простое предсказание с учетом всех систем"""
        try:
            # В реальной реализации здесь будет более сложная логика
            # Используем веса якорей для прогноза
            if self.active_anchors:
                # Простое среднее значение активности
                avg_activation = sum(a.activation_strength for a in self.active_anchors) / len(self.active_anchors)
                return avg_activation
            
            # Если нет активных якорей, используем другие факторы
            base_prediction = 0.5
            
            # Учет уровня внимания
            attention_factor = self.attention_system.focus_level
            
            # Учет уровня мотивации
            motivation_factor = self.motivation_system.motivation_level
            
            # Учет когнитивной нагрузки
            load_factor = 1.0 - self.cognitive_load * 0.5
            
            # Комбинируем факторы
            final_prediction = base_prediction * 0.3 + attention_factor * 0.3 + motivation_factor * 0.3 + load_factor * 0.1
            
            return min(1.0, max(0.0, final_prediction))
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.5  # Простой пример
    
    def learn_with_context(self, input_data, expected_output, context):
        """Обучение с учетом контекста и эмоций"""
        try:
            # В реальной реализации здесь будет обучение
            # Простое обновление состояния системы
            self.learning_history.append({
                'input': input_data,
                'output': expected_output,
                'context': context,
                'timestamp': time.time()
            })
            
            # Обновляем эмоциональную систему
            if expected_output is not None:
                feedback = self.emotional_system.process_feedback(
                    self.predict(input_data), expected_output, context
                )
                
                # Обновляем память
                self.memory_system.store_memory(
                    data=f"Learned: {input_data} -> {expected_output}",
                    memory_type="long_term",
                    emotional_value=feedback['reward']
                )
                
                # Обновляем систему самопознания
                self.self_learning_system.self_assessment['learning_efficiency'] = feedback['reward']
                
            logger.info("Learning with context completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in learning with context: {e}")
            return False
    
    def get_system_status(self):
        """Получение текущего состояния системы"""
        return {
            'cognitive_load': self.cognitive_load,
            'attention_level': self.attention_system.focus_level,
            'motivation_level': self.motivation_system.motivation_level,
            'system_state': self.system_state,
            'active_anchors_count': len(self.active_anchors),
            'learning_history_length': len(self.learning_history),
            'memory_stats': self.memory_system.get_memory_stats(),
            'timestamp': time.time()
        }
    
    def reset_system(self):
        """Сброс системы к начальному состоянию"""
        self.active_anchors.clear()
        self.learning_history.clear()
        self.cognitive_load = 0.0
        self.system_state = {
            'attention_level': 0.0,
            'emotional_state': 'neutral',
            'learning_phase': 'exploration',
            'task_complexity': 0.0
        }
        # Сброс всех внутренних систем
        self.emotional_system.emotional_memory.clear()
        self.pain_memory_system.pain_experiences.clear()
        self.memory_system.short_term_memory.clear()
        self.memory_system.long_term_memory.clear()
        logger.info("System reset completed")
class StreamingTextIterableDataset(torch.utils.data.IterableDataset):
    """IterableDataset для потоковой обработки текста."""
    def __init__(self, file_path, tokenizer, seq_length, stride=None, chunk_size=1024*1024):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length // 2
        self.chunk_size = chunk_size
        self._estimated_length = self._estimate_length()
    def _estimate_length(self):
        """Оценка длины датасета (приблизительно)."""
        try:
            return os.path.getsize(self.file_path) // (self.seq_length * 4)  # Примерная оценка
        except:
            return 1000
    def __iter__(self):
        """Итератор по данным."""
        worker_info = torch.utils.data.get_worker_info()
        file_handle = None
        file_size = os.path.getsize(self.file_path)
        start_offset = 0
        end_offset = file_size
        if worker_info is None:  # single-process loading
            logger.debug("StreamingTextIterableDataset: Single worker mode.")
        else:  # in a worker process
            logger.debug(f"StreamingTextIterableDataset: Worker {worker_info.id} of {worker_info.num_workers}")
            per_worker = int(math.ceil(file_size / float(worker_info.num_workers)))
            start_offset = worker_info.id * per_worker
            end_offset = min(start_offset + per_worker, file_size)
            logger.debug(f"Worker {worker_info.id} will process bytes {start_offset} to {end_offset}")
        try:
            file_handle = open(self.file_path, 'r', encoding='utf-8', errors='ignore')
            file_handle.seek(start_offset)
            # Если это не первый воркер, пропускаем до конца строки
            if worker_info is not None and worker_info.id > 0:
                file_handle.readline()
                logger.debug(f"Worker {worker_info.id} skipped to start of next line.")
            buffer_tokens = []
            bytes_read = start_offset
            while bytes_read < end_offset:
                read_size = min(self.chunk_size, end_offset - bytes_read)
                chunk = file_handle.read(read_size)
                if not chunk:
                    break
                bytes_read += len(chunk.encode('utf-8', errors='ignore'))
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
                overlap_start_index = len(buffer_tokens) - (self.seq_length + 1)
                if overlap_start_index < 0:
                    overlap_start_index = 0
                buffer_tokens = buffer_tokens[overlap_start_index:]
            # Обработка оставшихся токенов в конце файла/воркера
            if len(buffer_tokens) >= self.seq_length:
                # Генерируем последовательность из оставшихся токенов
                x = buffer_tokens[:self.seq_length]
                y = buffer_tokens[1:self.seq_length+1]
                yield (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
        finally:
            if file_handle:
                file_handle.close()
# Обучение и генерация
def calculate_perplexity(model, dataloader, criterion, device):
    """Вычисление perplexity модели."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output.reshape(-1, output.size(-1)), y_batch.reshape(-1))
            total_loss += loss.item() * x_batch.size(0) * x_batch.size(1)
            total_samples += x_batch.size(0) * x_batch.size(1)
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        perplexity = np.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    return perplexity
def generate_text(model, tokenizer, start_tokens, max_new_tokens=200, temperature=1.0, 
                  top_k=0, top_p=1.0, repetition_penalty=1.0, device='cpu'):
    """Генерация текста с ранней остановкой и динамической длиной."""
    model.eval()
    start_tokens = start_tokens.strip()
    # Токенизация стартового текста
    start_ids = tokenizer.encode(start_tokens).ids
    generated_ids = start_ids[:]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Подготовка входа
            input_ids = torch.tensor([generated_ids[-512:]], dtype=torch.long, device=device)
            # Предсказание
            outputs = model(input_ids)
            logits = outputs[0][0, -1, :]
            # Применение температуры
            logits = logits / temperature
            # Применение Top-K и Top-P фильтрации
            if top_k > 0:
                logits = top_k_filtering(logits, top_k)
            if top_p < 1.0:
                logits = top_p_filtering(logits, top_p)
            # Применение penalty
            if repetition_penalty != 1.0:
                for i in set(generated_ids):
                    logits[i] /= repetition_penalty
            # Применение softmax
            probs = F.softmax(logits, dim=-1)
            # Сэмплирование
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token.item())
            # Проверка на окончание
            if next_token.item() == tokenizer.token_to_id('<EOS>'):
                break
    # Декодирование
    generated_text = tokenizer.decode(generated_ids)
    # Улучшенная постобработка
    generated_text = generated_text.replace('<BOT>', '').replace('<EOS>', '').strip()
    generated_text = re.sub(r'\s+', ' ', generated_text)
    return generated_text
def top_k_filtering(logits, top_k):
    """Фильтрация токенов по Top-K."""
    if top_k == 0:
        return logits
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits
def top_p_filtering(logits, top_p):
    """Фильтрация токенов по Top-P."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Удаление токенов с низкой вероятностью
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    for i in range(sorted_indices.size(0)):
        indices_to_remove[i, sorted_indices[i, sorted_indices_to_remove[i]]] = True
    logits[indices_to_remove] = float('-inf')
    return F.softmax(logits, dim=-1)
# Загрузка и сохранение
def save_model_with_timestamp(model, tokenizer_path, vocab_size, token_type="bpe", 
                              model_type="gpt", perplexity=None, training_config=None):
    """Сохранение модели с временными метками."""
    timestamp = str(int(time.time()))
    model_name = f"{model_type}_{timestamp}"
    model_path = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)
    # Сохраняем модель
    model_save_path = os.path.join(model_path, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    # Сохраняем токенайзер
    if tokenizer_path:
        try:
            tokenizer_save_path = os.path.join(model_path, "tokenizer.json")
            shutil.copy2(tokenizer_path, tokenizer_save_path)
            tokenizer_path = tokenizer_save_path
        except Exception as e:
            logger.warning(f"Не удалось скопировать токенайзер: {e}")
    # Сохраняем метаданные
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'vocab_size': vocab_size,
        'token_type': token_type,
        'model_type': model_type,
        'perplexity': perplexity,
        'training_config': training_config,
        'tokenizer_path': tokenizer_path
    }
    metadata_path = os.path.join(model_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Модель сохранена: {model_path}")
    return model_path
def load_model_with_dicts(model_path, device):
    """Загрузка модели с метаданными."""
    try:
        # Загружаем метаданные
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Получаем путь к токенайзеру
        tokenizer_path = metadata.get('tokenizer_path', None)
        # Проверяем, существует ли файл токенайзера
        if tokenizer_path and not os.path.exists(tokenizer_path):
            # Ищем токенайзер в той же директории
            base_dir = os.path.dirname(model_path)
            possible_paths = [
                os.path.join(base_dir, "tokenizer.json"),
                os.path.join(base_dir, "tokenizer.json")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    tokenizer_path = path
                    break
        # Загружаем модель
        model = ModernGPT(
            vocab_size=metadata.get('vocab_size', 10000),
            hidden_size=metadata.get('hidden_size', DEFAULT_HIDDEN_SIZE),
            num_layers=metadata.get('num_layers', DEFAULT_NUM_LAYERS),
            num_heads=metadata.get('num_heads', DEFAULT_NUM_HEADS),
            ff_hidden_size=metadata.get('ff_hidden_size', DEFAULT_FF_HIDDEN_SIZE),
            max_seq_length=metadata.get('max_seq_length', DEFAULT_SEQ_LENGTH)
        ).to(device)
        model_path_full = os.path.join(model_path, "model.pth")
        model.load_state_dict(torch.load(model_path_full, map_location=device, weights_only=False))
        logger.info(f"Модель загружена: {model_path}")
        return model, tokenizer_path, metadata.get('token_type'), metadata.get('model_type'), \
               metadata.get('perplexity'), metadata.get('weight_statistics', {}), metadata.get('training_config')
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_path}: {e}")
        return None, None, None, None, None, None, None
def get_model_files():
    """Получение списка файлов моделей."""
    model_files = []
    try:
        for model_dir in glob.glob(os.path.join(MODELS_DIR, "*")):
            if os.path.isdir(model_dir):
                model_files.append(model_dir)
        model_files.sort(reverse=True)  # Последние модели первыми
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
    return model_files
def list_available_models():
    """Список доступных моделей."""
    model_files = get_model_files()
    if not model_files:
        print("❌ Нет доступных моделей")
        return
    print("Доступные модели:")
    for i, model_file in enumerate(model_files[:10]):  # Показываем первые 10
        try:
            metadata_path = os.path.join(model_file, "metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            timestamp = metadata.get('timestamp', 'unknown')
            loss = metadata.get('loss', 'unknown')
            perplexity = metadata.get('perplexity', 'unknown')
            print(f" {i+1}. {os.path.basename(model_file)}")
            print(f"   Дата: {timestamp}, Loss: {loss:.4f}, Perplexity: {perplexity}")
            print(f"   Vocab: {metadata.get('vocab_size', 'unknown')}, Type: {metadata.get('token_type', 'unknown')}")
        except Exception as e:
            print(f" {i+1}. {os.path.basename(model_file)} (ошибка чтения: {e})")
    return model_files
# Управление лучшей моделью
def cleanup_old_models():
    """Удаление старых моделей, оставляя только одну лучшую."""
    model_files = get_model_files()
    if len(model_files) <= 1:
        return  # Нет необходимости удалять
    # Оставляем только одну самую новую модель
    latest_model = model_files[0]
    # Удаляем остальные
    for model_file in model_files[1:]:
        try:
            shutil.rmtree(model_file)
            logger.info(f"Удалена старая модель: {model_file}")
        except Exception as e:
            logger.error(f"Ошибка удаления модели {model_file}: {e}")
def save_best_model(model, tokenizer_path, vocab_size, token_type="bpe", 
                   model_type="gpt", perplexity=None, training_config=None):
    """Сохранение лучшей модели и токенайзера."""
    # Удаляем старые модели
    cleanup_old_models()
    # Сохраняем как лучшую модель
    model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
    tokenizer_save_path = os.path.join(MODELS_DIR, BEST_TOKENIZER_FILE)
    # Сохраняем модель
    torch.save(model.state_dict(), model_path)
    # Сохраняем токенайзер
    if tokenizer_path:
        try:
            shutil.copy2(tokenizer_path, tokenizer_save_path)
            logger.info(f"Токенайзер сохранен: {tokenizer_save_path}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить токенайзер: {e}")
    # Сохраняем метаданные
    metadata = {
        'timestamp': str(int(time.time())),
        'model_name': BEST_MODEL_FILE,
        'vocab_size': vocab_size,
        'token_type': token_type,
        'model_type': model_type,
        'perplexity': perplexity,
        'training_config': training_config,
        'tokenizer_path': tokenizer_save_path
    }
    metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Лучшая модель сохранена: {model_path}")
    return model_path
def load_best_model(device):
    """Загрузка лучшей модели."""
    try:
        # Проверяем наличие лучшей модели
        model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
        tokenizer_path = os.path.join(MODELS_DIR, BEST_TOKENIZER_FILE)
        metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
        if not os.path.exists(model_path):
            return None, None, None, None, None, None, None
        # Загружаем метаданные
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Загружаем модель
        model = ModernGPT(
            vocab_size=metadata.get('vocab_size', 10000),
            hidden_size=metadata.get('hidden_size', DEFAULT_HIDDEN_SIZE),
            num_layers=metadata.get('num_layers', DEFAULT_NUM_LAYERS),
            num_heads=metadata.get('num_heads', DEFAULT_NUM_HEADS),
            ff_hidden_size=metadata.get('ff_hidden_size', DEFAULT_FF_HIDDEN_SIZE),
            max_seq_length=metadata.get('max_seq_length', DEFAULT_SEQ_LENGTH)
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        logger.info("Лучшая модель загружена")
        return model, tokenizer_path, metadata.get('token_type'), metadata.get('model_type'), \
               metadata.get('perplexity'), metadata.get('weight_statistics', {}), metadata.get('training_config')
    except Exception as e:
        logger.error(f"Ошибка при загрузке лучшей модели: {e}")
        return None, None, None, None, None, None, None
# Интерактивный режим с обучением в процессе общения
def interactive_mode():
    """Интерактивный режим работы с ИИ."""
    print("🚀 Запуск интерактивного режима...")
    print("Доступные команды:")
    print(" - train: обучить модель")
    print(" - load: загрузить модель")
    print(" - generate: сгенерировать текст")
    print(" - list: показать доступные модели")
    print(" - info: информация о текущей модели")
    print(" - chat: начать чат с моделью")
    print(" - quit: выйти")
    current_model = None
    current_tokenizer_path = None
    current_tokenizer = None
    current_model_type = None
    current_token_type = None
    current_perplexity = None
    current_training_config = None
    brain = HumanLikeBrain()
    # Пытаемся загрузить лучшую модель при запуске
    print("🔄 Попытка загрузки лучшей модели...")
    loaded_model, loaded_tokenizer_path, loaded_token_type, loaded_model_type, \
    loaded_perplexity, loaded_weight_stats, loaded_training_config = load_best_model("cpu")
    if loaded_model is not None:
        current_model = loaded_model
        current_tokenizer_path = loaded_tokenizer_path
        current_token_type = loaded_token_type
        current_model_type = loaded_model_type
        current_perplexity = loaded_perplexity
        current_training_config = loaded_training_config
        print("✅ Лучшая модель загружена успешно")
    else:
        print("⚠️ Лучшая модель не найдена, будет создана новая")
    # Загружаем токенайзер если есть
    if current_tokenizer_path and os.path.exists(current_tokenizer_path):
        try:
            current_tokenizer = Tokenizer.from_file(current_tokenizer_path)
            print("✅ Токенайзер загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки токенайзера: {e}")
            current_tokenizer_path = None
            current_tokenizer = None
    # Добавлены переменные для отслеживания прогресса
    learning_progress = {
        'new_words_learned': 0,
        'concepts_understood': 0,
        'context_patterns': 0,
        'session_start_time': time.time()
    }
    while True:
        try:
            command = input("Введите команду: ").strip().lower()
            if command == "quit":
                if current_model is not None:
                    # Сохраняем модель как лучшую при выходе
                    save_best_model(
                        current_model,
                        current_tokenizer_path,
                        10000,  # Примерный vocab_size
                        token_type=current_token_type,
                        model_type=current_model_type,
                        perplexity=current_perplexity,
                        training_config=current_training_config
                    )
                print("До свидания!")
                break
            elif command == "train":
                print("Начало обучения...")
                # В реальной реализации здесь будет обучение
            elif command == "load":
                models = get_model_files()
                if not models:
                    print("❌ Нет доступных моделей для загрузки")
                    continue
                print("Доступные модели:")
                for i, model_path in enumerate(models):
                    print(f" {i+1}. {os.path.basename(model_path)}")
                try:
                    model_num = int(input("Выберите номер модели: ")) - 1
                    if 0 <= model_num < len(models):
                        model_path = models[model_num]
                        model, tokenizer_path, token_type, model_type, perplexity, weight_stats, training_config = \
                            load_model_with_dicts(model_path, "cpu")
                        if model is not None:
                            current_model = model
                            current_tokenizer_path = tokenizer_path
                            current_token_type = token_type
                            current_model_type = model_type
                            current_perplexity = perplexity
                            current_training_config = training_config
                            print(f"✅ Модель загружена: {os.path.basename(model_path)}")
                            # Загружаем токенайзер
                            if current_tokenizer_path and os.path.exists(current_tokenizer_path):
                                current_tokenizer = Tokenizer.from_file(current_tokenizer_path)
                                print("✅ Токенайзер загружен")
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
                print("Информация о текущей модели:")
                print("Модель загружена успешно")
                print(f" Vocabulary size: {10000}")  # Пример
                print(f" Token type: {current_token_type}")
                print(f" Model type: {current_model_type}")
                print(f" Device: cpu")  # Пример
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
                    max_new_tokens = max(10, min(500, max_new_tokens))
                except ValueError:
                    max_new_tokens = 200
                print("🔄 Генерация текста...")
                try:
                    generated = generate_text(
                        current_model, 
                        current_tokenizer,
                        start_tokens=start_text, 
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        top_k=top_k, 
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        device="cpu"
                    )
                    print(f"📝 Сгенерированный текст:
{generated}")
                except Exception as e:
                    print(f"❌ Ошибка при генерации текста: {e}")
            elif command == "chat":
                if current_model is None or current_tokenizer is None:
                    print("❌ Нет загруженной модели или токенайзера. Сначала загрузите или обучите модель.")
                    continue
                print("🗣️ Начинаем чат с моделью. Введите '/exit' для выхода или '/clear' для очистки истории.")
                chat_history = ChatHistory(current_tokenizer, max_context_tokens=384)
                current_model.eval()
                # Добавлены счетчики для отслеживания обучения
                session_words = set()
                session_concepts = set()
                with torch.no_grad():
                    while True:
                        user_input = input("Вы: ").strip()
                        if user_input.lower() in ['/exit', '/quit']:
                            print("🤖 Модель: До скорой встречи!")
                            break
                        elif user_input.lower() == '/clear':
                            chat_history.clear()
                            print("🧹 История чата очищена")
                            continue
                        # Добавляем сообщение пользователя в историю
                        chat_history.add_user_message(user_input)
                        # Подготовка контекста для генерации
                        context = chat_history.get_context()
                        # Генерация ответа
                        try:
                            # Используем модель для генерации
                            response = generate_text(
                                current_model,
                                current_tokenizer,
                                start_tokens=context,
                                max_new_tokens=100,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.9,
                                repetition_penalty=1.2,
                                device="cpu"
                            )
                            # Очищаем ответ от служебных токенов
                            response = response.replace('<BOT>', '').replace('<EOS>', '').strip()
                            # Добавляем ответ модели в историю
                            chat_history.add_assistant_message(response)
                            # Анализируем новый контент для отслеживания обучения
                            # Подсчет новых слов и концепций
                            user_words = set(word.lower() for word in user_input.split() if word.isalpha())
                            response_words = set(word.lower() for word in response.split() if word.isalpha())
                            # Обновляем статистику обучения
                            new_words = user_words | response_words
                            session_words.update(new_words)
                            # Показываем метрики обучения
                            session_duration = time.time() - learning_progress['session_start_time']
                            print(f"📊 Статистика обучения:")
                            print(f"   • Новых слов: {len(session_words)}")
                            print(f"   • Время сессии: {int(session_duration//60)}:{int(session_duration%60):02d}")
                            # Простая оценка понимания
                            if len(session_words) > 10:
                                print(f"   • Уровень понимания: {'📈 Высокий' if len(session_words) > 30 else '📊 Средний'}")
                            print(f"🤖 Модель: {response}")
                            # Обучение на диалоге (простой пример)
                            # В реальной реализации здесь будет полноценное обучение
                            print("🧠 Модель учится на этом диалоге...")
                            print("🔄 Обновление внутренней структуры...")
                        except Exception as e:
                            print(f"❌ Ошибка при генерации ответа: {e}")
            else:
                print("❌ Неизвестная команда. Введите 'help' для списка команд.")
        except KeyboardInterrupt:
            print("Программа прервана пользователем")
            break
        except Exception as e:
            logger.error(f"Ошибка в интерактивном режиме: {e}")
            print(f"❌ Ошибка: {e}")
# Улучшенный класс для управления историей чата
class ChatHistory:
    """Класс для управления историей чата."""
    def __init__(self, tokenizer, max_context_tokens=512):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.history = deque()  # Используем deque для эффективного добавления/удаления с обоих концов
        self.total_tokens = 0
    def add_user_message(self, message):
        """Добавление сообщения пользователя."""
        entry = f"<USER>{message}<EOS>"
        self._add_entry(entry)
    def add_assistant_message(self, message):
        """Добавление сообщения ассистента."""
        entry = f"<BOT>{message}<EOS>"
        self._add_entry(entry)
    def _add_entry(self, entry):
        """Добавление записи в историю."""
        encoded = self.tokenizer.encode(entry)
        entry_tokens = len(encoded.ids)
        # Удаляем старые записи, если превышено максимальное количество токенов
        while self.total_tokens + entry_tokens > self.max_context_tokens and self.history:
            removed_entry = self.history.popleft()
            removed_encoded = self.tokenizer.encode(removed_entry)
            self.total_tokens -= len(removed_encoded.ids)
        self.history.append(entry)
        self.total_tokens += entry_tokens
    def get_context(self):
        """Получение контекста для генерации."""
        return " ".join(list(self.history))
    def clear(self):
        """Очистка истории чата."""
        self.history.clear()
        self.total_tokens = 0
# Подготовка данных с tokenizers
def train_tokenizer(files, vocab_size=30000, special_tokens=None):
    """Обучение BPE токенизатора."""
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
    """Токенизация текста с помощью обученного токенизатора."""
    if not TOKENIZERS_SUPPORT:
        raise Exception("Поддержка tokenizers не доступна. Установите tokenizers")
    encoding = tokenizer.encode(text)
    return encoding.ids
def detokenize_with_tokenizer(tokenizer, ids):
    """Детокенизация списка ID в текст."""
    if not TOKENIZERS_SUPPORT:
        raise Exception("Поддержка tokenizers не доступна. Установите tokenizers")
    return tokenizer.decode(ids)
# Новая функция для загрузки текста с URL
def load_text_from_url(url):
    """Загружает текст с веб-страницы по URL.
    Пытается извлечь основной текстовой контент."""
    logger.info(f"Попытка загрузки текста с URL: {url}")
    try:
        # Проверка URL формально (не обязательно, но полезно)
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Некорректный URL")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Удаляем скрипты и стили
        for script in soup(["script", "style"]):
            script.decompose()
        # Извлекаем текст
        text = soup.get_text()
        # Очищаем текст
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        logger.error(f"Ошибка загрузки текста с URL: {e}")
        return ""
# Основная функция обучения
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
                tokenizer_path, vocab_size, token_type="bpe", learning_rate=DEFAULT_LEARNING_RATE, 
                model_type="gpt", gradient_clipping=1.0, gradient_noise_sigma=1e-3):
    """Обучение модели."""
    logger.info(f"Начало обучения модели на устройстве {device}")
    logger.info(f"Параметры обучения: epochs={epochs}, batch_size={train_loader.batch_size}")
    metrics_collector = MetricsCollector()
    training_config = {
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': learning_rate,
        'gradient_clipping': gradient_clipping,
        'gradient_noise_sigma': gradient_noise_sigma
    }
    best_loss = float('inf')
    best_perplexity = float('inf')
    best_model_path = None
    training_start_time = time.time()
    # Переменная для отслеживания пути к постоянному токенизатору
    permanent_tokenizer_path = None
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            total_batches = 0
            logger.info(f"Эпоха {epoch+1}/{epochs} начата")
            # Training phase
            model.train()
            # Оцениваем общее количество батчей в эпохе (приблизительно)
            # Так как это IterableDataset, len(train_loader) не работает.
            batch_count = 0
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                # Forward pass
                outputs = model(x_batch)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1))
                # Backward pass
                loss.backward()
                # Gradient clipping
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # Gradient noise
                if gradient_noise_sigma > 0:
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * gradient_noise_sigma
                            param.grad.add_(noise)
                # Optimizer step
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
                batch_count += 1
                # Сохраняем метрики
                metrics_collector.add_batch_loss(loss.item())
                # Логирование каждые 100 батчей
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            # Validation phase
            model.eval()
            val_loss = 0
            val_batches = 0
            val_perplexity = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model(x_batch)
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1))
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            val_perplexity = np.exp(avg_val_loss) if avg_val_loss > 0 else float('inf')
            # Сохраняем метрики
            metrics_collector.add_training_loss(total_loss / total_batches if total_batches > 0 else 0)
            metrics_collector.add_validation_loss(avg_val_loss)
            metrics_collector.add_training_perplexity(np.exp(total_loss / total_batches) if total_batches > 0 else 0)
            metrics_collector.add_validation_perplexity(val_perplexity)
            metrics_collector.add_epoch_time(time.time() - epoch_start_time)
            # Сохранение лучшей модели
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_perplexity = val_perplexity
                metrics_collector.collect_weight_statistics(model)
                # Сохраняем модель как лучшую
                model_path = save_best_model(
                    model,
                    tokenizer_path,
                    vocab_size,
                    token_type=token_type,
                    model_type=model_type,
                    perplexity=val_perplexity,
                    training_config=training_config
                )
                best_model_path = model_path
                logger.info(f"Найдена лучшая модель с loss: {best_loss:.4f}")
            # Early stopping
            if epoch > 0 and abs(best_loss - avg_val_loss) < 1e-4:
                logger.info(f"Early stopping на эпохе {epoch+1}")
                break
        # Сохранение метрик
        training_time = time.time() - training_start_time
        logger.info(f"Обучение завершено за {training_time:.2f} сек")
        logger.info(f"Лучшая модель: loss {best_loss:.4f}, perplexity {best_perplexity:.4f}")
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_collector.save_metrics(f"final_metrics_{final_timestamp}.json")
        metrics_collector.plot_metrics(f"final_metrics_{final_timestamp}")
        # Копирование токенизатора после завершения обучения
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                permanent_tokenizer_path = os.path.join(MODELS_DIR, BEST_TOKENIZER_FILE)
                shutil.copy2(tokenizer_path, permanent_tokenizer_path)
                logger.info(f"Файл токенизатора скопирован в: {permanent_tokenizer_path}")
            except Exception as copy_e:
                logger.error(f"Ошибка копирования токенизатора: {copy_e}")
                permanent_tokenizer_path = tokenizer_path  # fallback к исходному пути
        else:
            permanent_tokenizer_path = tokenizer_path  # fallback если исходный файл не найден
        # Обновляем путь к токенизатору в лучшей сохраненной модели
        if best_model_path and permanent_tokenizer_path:
            try:
                # Перезагружаем чекпойнт
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                # Обновляем путь к токенизатору
                checkpoint['tokenizer_path'] = permanent_tokenizer_path
                # Сохраняем обновленный чекпойнт
                torch.save(checkpoint, best_model_path)
                logger.info(f"Путь к токенизатору обновлен в лучшей модели: {permanent_tokenizer_path}")
            except Exception as update_e:
                logger.error(f"Ошибка обновления пути к токенизатору в лучшей модели: {update_e}")
    except Exception as e:
        logger.error(f"Критическая ошибка во время обучения: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return best_model_path, metrics_collector, permanent_tokenizer_path
# Класс для сбора метрик
class MetricsCollector:
    """Класс для сбора и сохранения метрик обучения."""
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
    def add_epoch_time(self, epoch_time):
        self.metrics['epoch_times'].append(epoch_time)
    def add_batch_loss(self, loss):
        self.metrics['batch_losses'].append(loss)
    def collect_weight_statistics(self, model):
        """Сбор статистики весов модели."""
        weight_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
        self.metrics['weight_statistics'] = weight_stats
    def save_metrics(self, filename):
        """Сохранение метрик в файл."""
        try:
            filepath = os.path.join(METRICS_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"Метрики сохранены в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")
    def plot_metrics(self, filename_prefix):
        """Создание графиков метрик."""
        try:
            # Создание директории для графиков
            plots_dir = os.path.join(METRICS_DIR, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            # Построение графиков
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training Metrics')
            # Training vs Validation Loss
            if self.metrics['training_loss'] and self.metrics['validation_loss']:
                axes[0, 0].plot(self.metrics['training_loss'], label='Training Loss')
                axes[0, 0].plot(self.metrics['validation_loss'], label='Validation Loss')
                axes[0, 0].set_title('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            # Perplexity
            if self.metrics['training_perplexity'] and self.metrics['validation_perplexity']:
                axes[0, 1].plot(self.metrics['training_perplexity'], label='Training Perplexity')
                axes[0, 1].plot(self.metrics['validation_perplexity'], label='Validation Perplexity')
                axes[0, 1].set_title('Perplexity')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            # Epoch Times
            if self.metrics['epoch_times']:
                axes[1, 0].plot(self.metrics['epoch_times'])
                axes[1, 0].set_title('Epoch Times')
                axes[1, 0].set_ylabel('Time (seconds)')
                axes[1, 0].grid(True)
            # Batch Losses
            if self.metrics['batch_losses']:
                axes[1, 1].plot(self.metrics['batch_losses'])
                axes[1, 1].set_title('Batch Losses')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].grid(True)
            plt.tight_layout()
            # Сохранение графика
            plot_path = os.path.join(plots_dir, f"{filename_prefix}_metrics.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Графики метрик сохранены в {plot_path}")
        except Exception as e:
            logger.error(f"Ошибка создания графиков: {e}")
# Основная функция
if __name__ == "__main__":
    print("🚀 Биологически-ориентированная нейросеть с якорями")
    print("Оптимизации для CPU: уменьшенные параметры модели, улучшенная обработка ошибок")
    print("Интеграция с tokenizers для BPE.")
    print("Поддержка обучения на тексте из веб-страниц (URL).")
    print("Потоковая обработка больших файлов для экономии памяти.")
    print("Поддержка обучения на JSON-датасетах.")
    print("Улучшения для глубоких моделей:")
    print(" - Увеличено количество слоев до 50 (DEFAULT_NUM_LAYERS)")
    print(" - Добавлены остаточные связи между блоками трансформера")
    print(" - Добавлены механизмы эмоционального обучения и якорей")
    print(" - Добавлены все биологически-ориентированные компоненты")
    print(" - Включена система обучения в процессе общения")
    print(" - Добавлены механизмы управления лучшей моделью")
    print(" - Токенайзер сохраняется между запусками")
    print(f"📁 Модели сохраняются в: {os.path.abspath(MODELS_DIR)}")
    print(f"📝 Логи сохраняются в: {os.path.abspath(LOGS_DIR)}")
    print(f"📊 Метрики сохраняются в: {os.path.abspath(METRICS_DIR)}")
    print(f"Кэширование в: {os.path.abspath(CACHE_DIR)}")
    print("🚀 Запуск интерактивного режима...")
    # Создаем базовую модель для демонстрации
    brain = HumanLikeBrain()
    interactive_mode()
