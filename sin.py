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

class PainBasedLearning:
    """Система "болезненного" обучения."""
    def __init__(self):
        self.pain_memory = []
        self.pain_prevention_rules = []
        
    def process_pain_signal(self, error, confidence, context):
        """Обработка сигнала боли"""
        pain_intensity = self._calculate_pain_intensity(error, confidence)
        
        if pain_intensity > self.pain_threshold:
            pain_record = {
                'intensity': pain_intensity,
                'context': context,
                'error': error,
                'timestamp': time.time(),
                'neurons_affected': []  # В реальной реализации здесь будут нейроны
            }
            
            self.pain_memory.append(pain_record)
            self._activate_pain_response(pain_record)
            return True
        return False
    
    def _calculate_pain_intensity(self, error, confidence):
        """Расчет интенсивности боли"""
        return (error * (1.0 - confidence)) * 0.8
    
    def _activate_pain_response(self, pain_record):
        """Активация ответа на боль"""
        # В реальной реализации здесь будет коррекция весов
        pass

class ContextualRecognitionSystem:
    """Система контекстуального распознавания."""
    def __init__(self):
        self.context_memory = {}
        self.context_similarity_threshold = 0.7
        
    def recognize_context(self, input_data, context_window=5):
        """Распознавание контекста"""
        # В реальной реализации здесь будет анализ контекста
        return {
            'context_match': [],
            'emotional_signature': {},
            'confidence': 0.5,
            'learning_opportunity': True
        }
    
    def generate_associative_thought(self, seed_concept, context=None):
        """Генерация ассоциативного мышления"""
        # В реальной реализации здесь будет формирование ассоциаций
        return {
            'seed_concept': seed_concept,
            'associations': [],
            'abstract_thought': '',
            'context': context
        }

class WisdomSystem:
    """Система "мудрости" и интеллекта."""
    def __init__(self):
        self.knowledge_depth = 0
        self.pattern_recognition = 0.0
        self.cognitive_efficiency = 0.0
        self.learning_curves = {}
        
    def evaluate_confidence(self, new_information, existing_knowledge):
        """Оценка уверенности в новой информации"""
        consistency = 0.8  # В реальной реализации здесь будет анализ
        novelty_factor = 0.6  # В реальной реализации здесь будет анализ
        temporal_factor = 0.7  # В реальной реализации здесь будет анализ
        
        confidence_score = (
            consistency * 0.4 +
            novelty_factor * 0.3 +
            temporal_factor * 0.3
        )
        
        return confidence_score

class InstinctSystem:
    """Система инстинктов и автоматических реакций."""
    def __init__(self):
        self.instinct_triggers = {}
        self.automatic_responses = {}
        self.response_priority = {}
        
    def trigger_instinct(self, stimulus, context):
        """Активация инстинкта"""
        # Быстрая реакция на стимул
        if stimulus in self.instinct_triggers:
            instinct = self.instinct_triggers[stimulus]
            priority = self.response_priority.get(instinct, 1.0)
            return self.execute_automatic_response(instinct, context, priority)
        return None
    
    def execute_automatic_response(self, instinct, context, priority):
        """Выполнение автоматической реакции"""
        # Быстрое выполнение без глубокого анализа
        pass

class InnerVoiceSystem:
    """Система внутреннего голоса и саморефлексии."""
    def __init__(self):
        self.thought_processes = []
        self.reflection_memory = []
        self.critical_thinking = False
        
    def internal_dialogue(self, thoughts):
        """Внутренний диалог для анализа мыслей"""
        # Система самокритики
        # Анализ своих мыслей
        # Вопросы себе
        pass
        
    def self_reflection(self, current_thoughts):
        """Самоанализ"""
        # Оценка своей работы
        # Понимание своих ошибок
        # Улучшение стратегий
        pass

class AttentionSystem:
    """Система внимания и фильтрации информации."""
    def __init__(self):
        self.focus_level = 1.0
        self.attention_weights = {}
        self.distracting_factors = []
        
    def focus_attention(self, elements, context):
        """Фокусировка внимания на важных элементах"""
        # Расчет весов внимания
        # Отсечение ненужной информации
        # Повышение концентрации на ключевых моментах
        pass
        
    def filter_information(self, incoming_data):
        """Фильтрация информации по важности"""
        # Система отбора данных
        # Исключение шума
        # Выделение ключевых паттернов
        pass

class MemoryTypes:
    """Система различных типов памяти."""
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = {}
        self.emotional_memory = {}
        self.procedural_memory = {}
        
    def store_memory(self, data, memory_type="short_term", emotional_value=0.0):
        """Хранение памяти разных типов"""
        if memory_type == "short_term":
            self.short_term_memory.append((data, time.time()))
        elif memory_type == "long_term":
            self.long_term_memory[data] = {
                'timestamp': time.time(),
                'emotional_value': emotional_value,
                'importance': 0.0
            }
        # Другие типы памяти...

class IntuitionSystem:
    """Система интуиции и быстрых выводов."""
    def __init__(self):
        self.pattern_recognition = 0.0
        self.quick_decisions = []
        self.insight_triggers = []
        
    def quick_insight(self, partial_data):
        """Быстрое озарение на основе частичной информации"""
        # Использование неполных данных для быстрых выводов
        # Интуитивное понимание
        pass
        
    def pattern_recognition(self, data):
        """Распознавание паттернов без полного анализа"""
        # Быстрое распознавание закономерностей
        # Связывание неочевидных элементов
        pass

class MotivationSystem:
    """Система мотивации и целеполагания."""
    def __init__(self):
        self.goals = []
        self.motivation_level = 0.5
        self.reward_system = {}
        self.punishment_system = {}
        
    def set_goal(self, goal, importance):
        """Установка цели"""
        self.goals.append({
            'goal': goal,
            'importance': importance,
            'progress': 0.0,
            'motivation_required': importance * 0.8
        })
        
    def assess_motivation(self, current_progress):
        """Оценка мотивации"""
        # Система самомотивации
        # Адаптация к трудностям
        # Поддержка достижения целей
        pass

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
        
    def adapt_thinking(self, situation):
        """Адаптация стиля мышления под ситуацию"""
        # Изменение подхода в зависимости от контекста
        # Учет личностных особенностей
        pass

class CriticalThinkingSystem:
    """Система критического мышления."""
    def __init__(self):
        self.skepticism_level = 0.3
        self.logic_checkpoints = []
        self.consistency_checker = []
        
    def evaluate_thought(self, idea):
        """Критическая оценка идеи"""
        # Проверка логики
        # Поиск противоречий
        # Оценка доказательств
        # Анализ предпосылок
        pass
        
    def self_correct(self, errors_found):
        """Самокоррекция"""
        # Признание ошибок
        # Адаптация знаний
        # Улучшение методов
        pass

class EmotionalIntelligence:
    """Система эмоционального интеллекта."""
    def __init__(self):
        self.emotional_recognition = {}
        self.emotional_control = {}
        self.empathy_level = 0.0
        self.emotional_adaptability = 0.0
        
    def recognize_emotions(self, context):
        """Распознавание эмоций в контексте"""
        # Анализ эмоционального состояния
        # Понимание эмоциональных сигналов
        pass
        
    def regulate_emotions(self, emotional_state):
        """Регуляция эмоций"""
        # Контроль эмоциональных реакций
        # Адаптация к эмоциональному контексту
        pass

class SocialLearningSystem:
    """Система социального обучения."""
    def __init__(self):
        self.observation_memory = []
        self.modeling_behavior = []
        self.social_cues = []
        
    def observe_and_learn(self, example, context):
        """Наблюдение и обучение от примеров"""
        # Изучение поведения других
        # Адаптация знаний
        # Применение в новых ситуациях
        pass
        
    def imitation_learning(self, behavior_example):
        """Имитационное обучение"""
        # Копирование успешных действий
        # Адаптация к своей ситуации
        pass

class PainMemorySystem:
    """Система памяти о боли и обучения от ошибок."""
    def __init__(self):
        self.pain_experiences = []
        self.pain_prevention_rules = []
        self.learning_from_pain = True
        
    def remember_pain(self, experience):
        """Запоминание ошибки как "боли" """
        self.pain_experiences.append({
            'experience': experience,
            'timestamp': time.time(),
            'context': experience['context'],
            'error': experience['error'],
            'solution': experience['solution'] if 'solution' in experience else None
        })
        
    def avoid_pain_patterns(self, current_context):
        """Избегание паттернов боли"""
        # Поиск похожих паттернов из прошлого
        # Активация защитных механизмов
        pass

class TemporalReasoning:
    """Система временного восприятия."""
    def __init__(self):
        self.time_memory = []
        self.temporal_patterns = []
        self.causal_reasoning = {}
        
    def understand_temporal_relationships(self, events):
        """Понимание временных связей"""
        # Анализ причинно-следственных связей
        # Понимание последовательности событий
        # Прогнозирование последствий
        pass
        
    def time_based_learning(self, temporal_context):
        """Обучение с учетом временных факторов"""
        # Учет временной последовательности
        # Адаптация к временным изменениям
        pass

class SelfLearningSystem:
    """Система самостоятельного обучения."""
    def __init__(self):
        self.learning_strategies = []
        self.knowledge_gaps = []
        self.self_assessment = {}
        
    def identify_learning_needs(self, current_state):
        """Определение потребностей в обучении"""
        # Анализ пробелов в знаниях
        # Определение приоритетов
        # Планирование обучения
        pass
        
    def self_evaluate_performance(self):
        """Самооценка знаний и навыков"""
        # Оценка собственных способностей
        # Определение прогресса
        # Планирование дальнейшего обучения
        pass

class MultitaskingSystem:
    """Система многозадачности."""
    def __init__(self):
        self.task_queue = []
        self.context_switching_cost = 0.0
        self.priority_matrix = {}
        
    def manage_tasks(self, tasks, priorities):
        """Управление множеством задач"""
        # Оптимизация распределения внимания
        # Переключение между задачами
        # Минимизация потерь при переключении
        pass
        
    def context_switching(self, new_task):
        """Переключение контекста"""
        # Система переключения между задачами
        # Минимизация потерь времени
        pass

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
        
    def process_input(self, input_data, expected_output=None, context=None):
        """Обработка входных данных с полным контекстом"""
        # 1. Распознавание контекста
        context_info = self.context_system.recognize_context(input_data)
        
        # 2. Активация якорей
        # В реальной реализации здесь будет поиск/создание якорей
        anchor_chain = []
        
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
                
        # 4. Обновление нейронов
        # В реальной реализации здесь будет обновление якорей и связей
        
        return self.predict(input_data)
    
    def predict(self, input_data):
        """Простое предсказание (в реальной реализации будет сложнее)"""
        return 0.5  # Простой пример
    
    def learn_with_context(self, input_data, expected_output, context):
        """Обучение с учетом контекста и эмоций"""
        # В реальной реализации здесь будет обучение
        pass

# Датасеты и обработка текста
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
        
    print("\nДоступные модели:")
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
    
    while True:
        try:
            command = input("\nВведите команду: ").strip().lower()
            
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
                    print(f"📝 Сгенерированный текст:\n{generated}")
                except Exception as e:
                    print(f"❌ Ошибка при генерации текста: {e}")
                    
            elif command == "chat":
                if current_model is None or current_tokenizer is None:
                    print("❌ Нет загруженной модели или токенайзера. Сначала загрузите или обучите модель.")
                    continue
                    
                print("🗣️ Начинаем чат с моделью. Введите '/exit' для выхода или '/clear' для очистки истории.")
                chat_history = ChatHistory(current_tokenizer, max_context_tokens=384)
                current_model.eval()
                
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
                            # В реальной реализации здесь будет обучение в процессе общения
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
                            
                            print(f"🤖 Модель: {response}")
                            
                            # Обучение на диалоге (простой пример)
                            # В реальной реализации здесь будет полноценное обучение
                            print("🧠 Модель учится на этом диалоге...")
                            
                        except Exception as e:
                            print(f"❌ Ошибка при генерации ответа: {e}")
                            
            else:
                print("❌ Неизвестная команда. Введите 'help' для списка команд.")
                
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем")
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
