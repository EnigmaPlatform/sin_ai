import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from collections import deque
import gradio as gr
from llama_cpp import Llama
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
import logging
from typing import List, Dict, Union, Optional, Tuple, Any, Generator
from huggingface_hub import hf_hub_download, snapshot_download
import random
import time
from enum import Enum
import hashlib
from datetime import datetime, timedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import html
from threading import Lock, Thread
from queue import Queue
import asyncio
import aiohttp
from functools import wraps
import uuid
import gzip
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import signal
import math
from pathlib import Path

# --- Настройка логгирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Оптимизированная конфигурация для Tecno Pova 5 ---
CONFIG = {
    "MODEL_REPO": "TheBloke/MobileLLaMA-1.4B-Chat-GGUF",
    "MODEL_FILE": "mobilellama-1.4b-chat.Q4_K_M.gguf",
    "LOCAL_MODEL_DIR": "/storage/emulated/0/Download/models",
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "MAX_HISTORY": 5,
    "WHOOSH_INDEX_DIR": "whoosh_index",
    "CHROMA_DB_PATH": "chroma_db",
    "MODEL_CACHE_DIR": "/storage/emulated/0/Download/models_cache",
    "DEVICE": "cpu",
    "LOCAL_FILES_ONLY": False,
    "HF_HOME": "/storage/emulated/0/Download/models_cache",
    "EMBEDDING_DIM": 384,
    "LLM_CTX_SIZE": 1024,
    "LLM_THREADS": 2,
    "MOOD_UPDATE_INTERVAL": 60,
    "MOOD_DECAY_RATE": 0.95,
    "MOOD_VOLATILITY": 0.1,
    "MEMORY_DECAY_RATE": 0.98,
    "LONG_TERM_MEMORY_SIZE": 100,
    "MAX_RESPONSE_TOKENS": 100,
    "TOP_P": 0.9,
    "TEMPERATURE_RANGE": (0.5, 1.0),
    "MEMORY_IMPORTANCE_THRESHOLD": 0.3,
    "MAX_MEMORY_RETRIEVAL": 3,
    "RATE_LIMIT": 3,
    "MAX_SESSION_AGE": 86400,
    "STREAMING_CHUNK_SIZE": 30,
    "MAX_INPUT_LENGTH": 500,
    "COMPRESS_MEMORY": True,
    "THREAD_POOL_SIZE": 2,
    "PERSONA": {
        "name": "Син",
        "age": 20,
        "appearance": "Невысокая, хрупкого телосложения, короткие растрёпанные каштановые волосы с рыжеватыми прядями, большие зелёные глаза, часто носит свободные худи",
        "core_traits": ["озорная", "чуткая", "контрастная", "игривая", "ранимая"],
        "speech_style": "эмоциональная, с использованием разговорных и игривых выражений",
        "favorite_topics": ["искусство", "психология", "технологии", "музыка"],
        "disliked_topics": ["политика", "насилие"],
        "relationship_development": {
            "positive_increment": 0.05,
            "negative_decrement": 0.1,
            "base_level": 0.5
        }
    },
    "REFUSAL_RESPONSES": [
        "Я не хочу сейчас об этом говорить...",
        "Может, поговорим о чем-то другом?",
        "Я не в настроении обсуждать это.",
        "Прости, но я не готова продолжать этот разговор.",
        "Давай сделаем перерыв, хорошо?",
        "Я чувствую себя некомфортно с этой темой.",
        "Мне нужно время, чтобы подумать об этом.",
        "Может, обсудим что-то более приятное?",
        "Я не уверена, что хочу продолжать...",
        "*молчит, отворачивается*"
    ],
    "ANGRY_RESPONSES": [
        "Я не хочу больше разговаривать! Оставь меня в покое!",
        "Хватит! Я не буду это обсуждать!",
        "Ты меня совсем разозлил! Я ухожу!",
        "*резко встает и уходит*",
        "Нет, всё! Разговор окончен!",
        "Я слишком зла, чтобы продолжать!",
        "Ты перешел границы! Прощай!",
        "*закрывает лицо руками* Уйди!",
        "Система: Син слишком расстроена и просит оставить её одну. Попробуйте позже.",
        "*игнорирует сообщения*"
    ],
    "EMOTION_TRIGGERS": {
        "positive": {
            "words": ["спасибо", "умничка", "хорошо", "прекрасно", "нравится", "люблю", "восхитительно", "великолепно", "замечательно"],
            "mood_change": 15,
            "anger_change": -10,
            "relationship_change": 5,
            "possible_emotions": ["JOY", "EXCITEMENT", "CONTENTMENT"]
        },
        "negative": {
            "words": ["тупой", "глупо", "плохо", "ужасно", "ненавижу", "разочарован", "отвратительно", "бесит"],
            "mood_change": -25,
            "anger_change": 20,
            "relationship_change": -10,
            "possible_emotions": ["FRUSTRATION", "SADNESS", "ANNOYANCE"]
        },
        "anger": {
            "words": ["злит", "бесит", "раздражает", "идиот", "дурак", "ненавижу"],
            "mood_change": -30,
            "anger_change": 30,
            "relationship_change": -15,
            "possible_emotions": ["ANGER"]
        },
        "sadness": {
            "words": ["грустно", "печально", "плакать", "несчастный", "тоска", "одиночество"],
            "mood_change": -20,
            "anger_change": 5,
            "relationship_change": 5,
            "possible_emotions": ["SADNESS"]
        },
        "excitement": {
            "words": ["вау", "круто", "потрясающе", "восхитительно", "невероятно", "удивительно"],
            "mood_change": 20,
            "anger_change": -5,
            "relationship_change": 5,
            "possible_emotions": ["EXCITEMENT"]
        }
    }
}

# Создание директорий
for dir_key in ["LOCAL_MODEL_DIR", "WHOOSH_INDEX_DIR", "CHROMA_DB_PATH", "MODEL_CACHE_DIR"]:
    os.makedirs(CONFIG[dir_key], exist_ok=True)

# --- Вспомогательные функции ---
def rate_limited(max_per_second: int):
    """Декоратор для ограничения частоты запросов"""
    lock = Lock()
    min_interval = 1.0 / max_per_second

    def decorate(func):
        last_time_called = time.perf_counter()

        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            nonlocal last_time_called
            with lock:
                elapsed = time.perf_counter() - last_time_called
                wait_time = min_interval - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                last_time_called = time.perf_counter()
            return func(*args, **kwargs)
        return rate_limited_function
    return decorate

def validate_input(text: str) -> Tuple[bool, str]:
    """Валидация пользовательского ввода"""
    if not text or len(text.strip()) == 0:
        return False, "Запрос не может быть пустым"
    
    if len(text) > CONFIG["MAX_INPUT_LENGTH"]:
        return False, f"Запрос слишком длинный (максимум {CONFIG['MAX_INPUT_LENGTH']} символов)"
    
    if re.search(r'[\x00-\x1F\x7F-\x9F]', text):
        return False, "Запрос содержит недопустимые символы"
    
    return True, ""

def compress_data(data: Any) -> bytes:
    """Сжатие данных для хранения в памяти"""
    if not CONFIG["COMPRESS_MEMORY"]:
        return pickle.dumps(data)
    return gzip.compress(pickle.dumps(data))

def decompress_data(data: bytes) -> Any:
    """Распаковка данных из памяти"""
    if not CONFIG["COMPRESS_MEMORY"]:
        return pickle.loads(data)
    return pickle.loads(gzip.decompress(data))

def download_model(repo_id: str, filename: str, local_dir: str) -> str:
    """Загрузка модели с обработкой ошибок"""
    model_path = os.path.join(local_dir, filename)
    
    if os.path.exists(model_path):
        logger.info(f"Модель {filename} уже существует локально")
        return model_path
    
    logger.info(f"Начинаем загрузку модели {filename}...")
    
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=local_dir,
            resume_download=True,
            local_files_only=False
        )
        
        logger.info(f"Модель успешно загружена в {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise

# --- Классы ---
class EmotionState(Enum):
    JOY = "радость"
    ANGER = "злость"
    SADNESS = "грусть"
    NEUTRAL = "нейтральное"
    EXCITEMENT = "волнение"
    CONFUSION = "замешательство"
    FRUSTRATION = "разочарование"
    CURIOSITY = "любопытство"
    SHYNESS = "застенчивость"
    PLAYFULNESS = "озорное настроение"
    ANNOYANCE = "раздражение"
    FLIRTATION = "игриво-кокетливое"
    NOSTALGIA = "ностальгия"
    CONTENTMENT = "удовлетворение"
    AMUSEMENT = "веселье"
    EMBARRASSMENT = "смущение"
    ANTICIPATION = "предвкушение"

class EmotionalState:
    def __init__(self):
        self.mood = 50  # 0-100
        self.emotion = EmotionState.NEUTRAL
        self.emotion_intensity = 0.5
        self.last_update_time = time.time()
        self.emotion_history = deque(maxlen=20)
        self.stability = 0.7
        self.character_traits = {
            "playfulness": 0.8,
            "shyness": 0.6,
            "sensitivity": 0.7,
            "temper": 0.4,
            "optimism": 0.5,
            "empathy": 0.6
        }
        self.recent_actions = deque(maxlen=5)
        self.anger_level = 0  # 0-100
        self.relationship_level = 50  # 0-100
        self.fatigue = 0  # 0-100
        self.lock = Lock()
        self.long_term_mood = 50
        self.mood_volatility = CONFIG["MOOD_VOLATILITY"]
        self.trait_development = {
            "playfulness": 0.01,
            "shyness": 0.005,
            "sensitivity": 0.008,
            "temper": 0.003
        }
        
    def update(self, text: str = None, user_action: str = None):
        with self.lock:
            now = time.time()
            time_elapsed = now - self.last_update_time
            
            if user_action:
                self.recent_actions.append(user_action)
            
            if time_elapsed > CONFIG["MOOD_UPDATE_INTERVAL"]:
                self._apply_decay(time_elapsed)
                self._apply_random_fluctuations()
                self._develop_traits(time_elapsed)
                self.last_update_time = now
            
            if text:
                self._update_from_text(text)
            
            self._update_fatigue()
            self._determine_emotion_state()
            
            logger.debug(f"Состояние: {self.emotion.name} (интенсивность: {self.emotion_intensity:.2f}), настроение: {self.mood}, злость: {self.anger_level}, отношения: {self.relationship_level}")
    
    def _apply_decay(self, time_elapsed: float):
        decay_factor = CONFIG["MOOD_DECAY_RATE"] ** (time_elapsed / CONFIG["MOOD_UPDATE_INTERVAL"])
        self.mood = 50 + (self.mood - 50) * decay_factor
        self.emotion_intensity *= decay_factor
        self.anger_level = max(0, self.anger_level * decay_factor)
        self.relationship_level = 50 + (self.relationship_level - 50) * decay_factor
        self.long_term_mood = 50 + (self.long_term_mood - 50) * (decay_factor ** 0.5)
        
    def _apply_random_fluctuations(self):
        playfulness_effect = random.gauss(0, self.character_traits["playfulness"] * 0.1)
        shyness_effect = -random.gauss(0, self.character_traits["shyness"] * 0.05)
        optimism_effect = random.gauss(0, self.character_traits["optimism"] * 0.07)
        
        volatility = self.mood_volatility * (1 + 0.5 * math.sin(time.time() / 86400))
        
        fluctuation = random.gauss(0, volatility * 10) + playfulness_effect + shyness_effect + optimism_effect
        self.mood = max(0, min(100, self.mood + fluctuation))
        
        intensity_change = random.gauss(0, CONFIG["MOOD_DECAY_RATE"] * 0.15)
        self.emotion_intensity = max(0, min(1, self.emotion_intensity + intensity_change))
        
        self.long_term_mood = max(0, min(100, self.long_term_mood + random.gauss(0, volatility * 2)))
    
    def _develop_traits(self, time_elapsed: float):
        for trait, change_rate in self.trait_development.items():
            trait_emotions = {
                "playfulness": [EmotionState.PLAYFULNESS, EmotionState.AMUSEMENT],
                "shyness": [EmotionState.SHYNESS, EmotionState.EMBARRASSMENT],
                "sensitivity": [EmotionState.SADNESS, EmotionState.JOY],
                "temper": [EmotionState.ANGER, EmotionState.ANNOYANCE]
            }
            
            if trait in trait_emotions:
                emotion_count = sum(1 for e in self.emotion_history 
                                  if e["emotion"] in trait_emotions[trait])
                total = len(self.emotion_history)
                if total > 0:
                    emotion_ratio = emotion_count / total
                    change = change_rate * (emotion_ratio - 0.3)
                    self.character_traits[trait] = max(0, min(1, self.character_traits[trait] + change))
    
    def _update_from_text(self, text: str):
        text_lower = text.lower()
        mood_change = 0
        emotion_shift = None
        anger_change = 0
        relationship_change = 0
        
        for category, triggers in CONFIG["EMOTION_TRIGGERS"].items():
            if any(word in text_lower for word in triggers["words"]):
                mood_change += triggers["mood_change"] * self.stability
                anger_change += triggers["anger_change"]
                relationship_change += triggers["relationship_change"]
                emotion_shift = random.choice([EmotionState[e] for e in triggers["possible_emotions"]])
                break
        
        if any(topic.lower() in text_lower for topic in CONFIG["PERSONA"]["favorite_topics"]):
            mood_change += 15 * self.stability
            relationship_change += 5
            if not emotion_shift:
                emotion_shift = EmotionState.EXCITEMENT
        elif any(topic.lower() in text_lower for topic in CONFIG["PERSONA"]["disliked_topics"]):
            mood_change -= 20 * self.stability
            anger_change += 15
            relationship_change -= 10
            if not emotion_shift:
                emotion_shift = EmotionState.ANGER
        
        if any(q in text_lower for q in ["как дела", "как настроение", "как жизнь"]):
            mood_change += 10 * self.stability
            relationship_change += 8
            if not emotion_shift:
                emotion_shift = EmotionState.CURIOSITY
        
        self._apply_emotional_changes(mood_change, anger_change, relationship_change, emotion_shift)
        
        self.emotion_history.append({
            "emotion": self.emotion,
            "intensity": self.emotion_intensity,
            "mood": self.mood,
            "anger": self.anger_level,
            "relationship": self.relationship_level,
            "timestamp": time.time()
        })
    
    def _apply_emotional_changes(self, mood_change: float, anger_change: float, 
                               relationship_change: float, emotion_shift: Optional[EmotionState]):
        self.mood = max(0, min(100, self.mood + mood_change))
        self.anger_level = max(0, min(100, self.anger_level + anger_change))
        self.relationship_level = max(0, min(100, self.relationship_level + relationship_change))
        
        if emotion_shift:
            if self.emotion == emotion_shift:
                self.emotion_intensity = min(1, self.emotion_intensity + 0.2)
            else:
                self.emotion_intensity = max(0.3, self.emotion_intensity - 0.1)
                if self.emotion_intensity <= 0.3:
                    self.emotion = emotion_shift
                    self.emotion_intensity = 0.7
    
    def _update_fatigue(self):
        self.fatigue = min(100, self.fatigue + 1 + self.emotion_intensity * 0.5)
        
        recovery_rate = 0.05
        if self.emotion in [EmotionState.NEUTRAL, EmotionState.CONTENTMENT]:
            recovery_rate = 0.1
        elif self.emotion in [EmotionState.ANGER, EmotionState.EXCITEMENT]:
            recovery_rate = 0.02
            
        if random.random() < recovery_rate:
            self.fatigue = max(0, self.fatigue - (3 + random.random() * 4))
    
    def _determine_emotion_state(self):
        self._apply_trait_effects()
        self._apply_mood_effects()
        self._apply_special_conditions()
        self.emotion_intensity = max(0.1, min(1.0, self.emotion_intensity))
    
    def _apply_trait_effects(self):
        if random.random() < self.character_traits["playfulness"]/8 and self.emotion != EmotionState.ANGER:
            self.emotion = EmotionState.PLAYFULNESS
            self.emotion_intensity = min(1, self.emotion_intensity + 0.15)
        
        if random.random() < self.character_traits["shyness"]/8 and self.emotion_intensity > 0.4:
            self.emotion = EmotionState.SHYNESS
            self.emotion_intensity = min(1, self.emotion_intensity * 1.15)
        
        if self.character_traits["sensitivity"] > 0.6 and self.emotion_intensity > 0.4:
            self.emotion_intensity = min(1, self.emotion_intensity * 1.1)
    
    def _apply_mood_effects(self):
        if self.mood > 75:
            if self.emotion not in [EmotionState.JOY, EmotionState.EXCITEMENT, EmotionState.PLAYFULNESS]:
                self.emotion = random.choice([EmotionState.JOY, EmotionState.EXCITEMENT, EmotionState.CONTENTMENT])
        elif self.mood < 25:
            if self.emotion not in [EmotionState.SADNESS, EmotionState.ANGER, EmotionState.FRUSTRATION]:
                self.emotion = random.choice([EmotionState.SADNESS, EmotionState.ANGER, EmotionState.FRUSTRATION])
    
    def _apply_special_conditions(self):
        if self.anger_level > 70:
            self.emotion = EmotionState.ANGER
            self.emotion_intensity = max(0.8, self.emotion_intensity)
        elif self.anger_level > 50:
            self.emotion = EmotionState.ANNOYANCE
            self.emotion_intensity = max(0.6, self.emotion_intensity)
        
        if self.fatigue > 70 and self.emotion_intensity > 0.4:
            self.emotion = random.choice([EmotionState.FRUSTRATION, EmotionState.SADNESS])
            self.emotion_intensity = min(1, self.emotion_intensity + 0.15)
        
        if self.relationship_level > 70 and random.random() < 0.3:
            self.emotion = random.choice([EmotionState.PLAYFULNESS, EmotionState.FLIRTATION])
    
    def get_temperature(self) -> float:
        base_temp = 0.7
        intensity = self.emotion_intensity
        
        temp_adjustments = {
            EmotionState.JOY: 0.2,
            EmotionState.ANGER: 0.4,
            EmotionState.EXCITEMENT: 0.3,
            EmotionState.SADNESS: -0.2,
            EmotionState.FRUSTRATION: 0.2,
            EmotionState.CURIOSITY: 0.1,
            EmotionState.PLAYFULNESS: 0.5,
            EmotionState.SHYNESS: -0.2,
            EmotionState.ANNOYANCE: 0.3,
            EmotionState.FLIRTATION: 0.4
        }
        
        adjustment = temp_adjustments.get(self.emotion, 0)
        temperature = base_temp + adjustment * intensity
        return min(max(CONFIG["TEMPERATURE_RANGE"][0], temperature), CONFIG["TEMPERATURE_RANGE"][1])
    
    def get_emotional_description(self) -> str:
        intensity_desc = ""
        if self.emotion_intensity > 0.8:
            intensity_desc = "очень "
        elif self.emotion_intensity > 0.5:
            intensity_desc = ""
        elif self.emotion_intensity > 0.3:
            intensity_desc = "слегка "
        else:
            intensity_desc = "почти не "
        
        desc = f"{intensity_desc}{self.emotion.value}"
        
        if self.emotion == EmotionState.PLAYFULNESS:
            play_level = "очень" if self.emotion_intensity > 0.7 else ""
            desc = f"{play_level} озорное настроение"
        elif self.emotion == EmotionState.SHYNESS:
            shy_level = "крайне" if self.emotion_intensity > 0.7 else ""
            desc = f"{shy_level} застенчивая"
        elif self.emotion == EmotionState.ANGER and self.emotion_intensity > 0.8:
            desc = "в ярости"
        elif self.emotion == EmotionState.FLIRTATION:
            desc = "игриво-кокетливое настроение"
        
        if self.relationship_level > 70:
            desc += ", расположена к вам"
        elif self.relationship_level < 30:
            desc += ", насторожена"
        
        if self.fatigue > 70:
            desc += ", устала"
        
        return desc
    
    def get_emoji(self) -> str:
        emoji_map = {
            EmotionState.JOY: "😊",
            EmotionState.ANGER: "😠",
            EmotionState.SADNESS: "😢",
            EmotionState.NEUTRAL: "😐",
            EmotionState.EXCITEMENT: "🤩",
            EmotionState.CONFUSION: "😕",
            EmotionState.FRUSTRATION: "😤",
            EmotionState.CURIOSITY: "🤔",
            EmotionState.PLAYFULNESS: "😜",
            EmotionState.SHYNESS: "🥺",
            EmotionState.ANNOYANCE: "😒",
            EmotionState.FLIRTATION: "😳",
            EmotionState.NOSTALGIA: "😌",
            EmotionState.CONTENTMENT: "😌",
            EmotionState.AMUSEMENT: "😂",
            EmotionState.EMBARRASSMENT: "😳",
            EmotionState.ANTICIPATION: "🤩"
        }
        return emoji_map.get(self.emotion, "❓")
    
    def get_character_modifiers(self) -> str:
        modifiers = []
        
        if self.emotion == EmotionState.PLAYFULNESS:
            modifiers.append("игривый тон")
            if random.random() < self.character_traits["playfulness"]:
                modifiers.append("лёгкая дразнилка")
        
        if self.emotion == EmotionState.SHYNESS:
            modifiers.append("робкий тон")
            if random.random() < self.character_traits["shyness"]:
                modifiers.append("смущение")
        
        if self.character_traits["sensitivity"] > 0.6 and self.emotion_intensity > 0.5:
            modifiers.append("эмоциональный отклик")
        
        if self.anger_level > 50:
            modifiers.append("раздражение")
        
        if self.fatigue > 60:
            modifiers.append("усталость")
        
        if self.relationship_level > 70:
            modifiers.append("доверие")
        elif self.relationship_level < 30:
            modifiers.append("недоверие")
        
        return ", ".join(modifiers) if modifiers else "стандартный"
    
    def is_too_angry(self) -> bool:
        return self.anger_level > 80
    
    def is_refusing_to_talk(self) -> bool:
        return (self.anger_level > 70 or 
                (self.emotion == EmotionState.SADNESS and self.emotion_intensity > 0.7) or
                self.fatigue > 85)

class MemoryItem:
    def __init__(self, content: str, importance: float = 0.5, memory_type: str = "fact"):
        self.content = content
        self.importance = max(0, min(1, importance))
        self.memory_type = memory_type
        self.creation_time = time.time()
        self.last_access_time = time.time()
        self.access_count = 0
        self.embedding = None
        self.lock = Lock()
    
    def decay(self):
        with self.lock:
            time_passed = time.time() - self.last_access_time
            decay_factor = CONFIG["MEMORY_DECAY_RATE"] ** (time_passed / (3600 * 24))
            self.importance *= decay_factor
    
    def access(self):
        with self.lock:
            self.last_access_time = time.time()
            self.access_count += 1
            self.importance = min(1, self.importance + 0.05)
    
    def __str__(self):
        return f"{self.content} (важность: {self.importance:.2f}, тип: {self.memory_type})"

class LongTermMemory:
    def __init__(self, max_size: int = 100):
        self.memories = []
        self.max_size = max_size
        self.lock = Lock()
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer(
                CONFIG["EMBEDDING_MODEL"],
                cache_folder=CONFIG["MODEL_CACHE_DIR"],
                device=CONFIG["DEVICE"]
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации модели эмбеддингов для памяти: {e}")
            raise
    
    def add(self, content: str, importance: float = 0.5, memory_type: str = "fact"):
        with self.lock:
            if len(self.memories) >= self.max_size:
                self.memories.sort(key=lambda x: x.importance)
                self.memories = self.memories[len(self.memories)//2:]
            
            new_memory = MemoryItem(content, importance, memory_type)
            
            try:
                embedding = self.embedding_model.encode(content)
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.numpy()
                new_memory.embedding = embedding
            except Exception as e:
                logger.error(f"Ошибка генерации эмбеддинга для памяти: {e}")
                new_memory.embedding = None
            
            self.memories.append(new_memory)
    
    def retrieve(self, query: str = None, n: int = 3) -> List[MemoryItem]:
        with self.lock:
            for memory in self.memories:
                memory.decay()
            
            if not query:
                sorted_memories = sorted(self.memories, key=lambda x: -x.importance)
                return sorted_memories[:n]
            
            try:
                query_embedding = self.embedding_model.encode(query)
                if isinstance(query_embedding, torch.Tensor):
                    query_embedding = query_embedding.numpy()
                
                similarities = []
                for memory in self.memories:
                    if memory.embedding is not None:
                        sim = cosine_similarity(
                            [query_embedding],
                            [memory.embedding]
                        )[0][0] * memory.importance
                        similarities.append((sim, memory))
                
                similarities.sort(key=lambda x: -x[0])
                return [mem for sim, mem in similarities[:n]]
            except Exception as e:
                logger.error(f"Ошибка поиска в долговременной памяти: {e}")
                return sorted(self.memories, key=lambda x: -x.importance)[:n]
    
    def get_related_memories(self, topic: str) -> List[MemoryItem]:
        with self.lock:
            related = self.retrieve(topic, CONFIG["MAX_MEMORY_RETRIEVAL"])
            for memory in related:
                memory.access()
            return related
    
    def save(self, filepath: str):
        with self.lock:
            try:
                with open(filepath, 'wb') as f:
                    memories_to_save = []
                    for mem in self.memories:
                        mem_copy = MemoryItem(mem.content, mem.importance, mem.memory_type)
                        mem_copy.creation_time = mem.creation_time
                        mem_copy.last_access_time = mem.last_access_time
                        mem_copy.access_count = mem.access_count
                        memories_to_save.append(mem_copy)
                    
                    pickle.dump(memories_to_save, f)
            except Exception as e:
                logger.error(f"Ошибка сохранения памяти: {e}")
                raise
    
    def load(self, filepath: str):
        with self.lock:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        self.memories = pickle.load(f)
                        
                        for mem in self.memories:
                            try:
                                embedding = self.embedding_model.encode(mem.content)
                                if isinstance(embedding, torch.Tensor):
                                    embedding = embedding.numpy()
                                mem.embedding = embedding
                            except Exception as e:
                                logger.error(f"Ошибка генерации эмбеддинга при загрузке памяти: {e}")
                                mem.embedding = None
                except Exception as e:
                    logger.error(f"Ошибка загрузки памяти: {e}")
                    self.memories = []

@dataclass
class SessionData:
    session_id: str
    history: deque
    emotional_state: EmotionalState
    long_term_memory: LongTermMemory
    created_at: float
    last_accessed: float

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG["THREAD_POOL_SIZE"])
        
    def create_session(self) -> SessionData:
        session_id = str(uuid.uuid4())
        session = SessionData(
            session_id=session_id,
            history=deque(maxlen=CONFIG["MAX_HISTORY"]),
            emotional_state=EmotionalState(),
            long_term_memory=LongTermMemory(CONFIG["LONG_TERM_MEMORY_SIZE"]),
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        with self.lock:
            self.sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = time.time()
            return session
    
    def cleanup_sessions(self):
        with self.lock:
            current_time = time.time()
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if current_time - session.last_accessed > CONFIG["MAX_SESSION_AGE"]
            ]
            
            for sid in expired_sessions:
                session = self.sessions.pop(sid)
                self._save_session_data(session)
                logger.info(f"Сессия {sid} очищена")
    
    def _save_session_data(self, session: SessionData):
        try:
            if session.long_term_memory:
                memory_file = os.path.join(CONFIG["CHROMA_DB_PATH"], f"memory_{session.session_id}.pkl")
                session.long_term_memory.save(memory_file)
        except Exception as e:
            logger.error(f"Ошибка сохранения данных сессии {session.session_id}: {e}")
    
    def save_all_sessions(self):
        with self.lock:
            for session in self.sessions.values():
                self._save_session_data(session)
    
    def __del__(self):
        self.save_all_sessions()
        self.executor.shutdown(wait=True)

class Assistant:
    def __init__(self):
        logger.info("Инициализация ассистента для Tecno Pova 5...")
        self.session_manager = SessionManager()
        self._initialize_models()
        self._initialize_databases()
        self.rate_limit_cache = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG["THREAD_POOL_SIZE"])
        self._start_cleanup_task()
        logger.info("Ассистент успешно инициализирован для мобильного устройства")
    
    def _start_cleanup_task(self):
        def cleanup():
            while True:
                time.sleep(3600)
                self.session_manager.cleanup_sessions()
        
        Thread(target=cleanup, daemon=True).start()
    
    def _initialize_models(self):
        try:
            logger.info("Загрузка облегченной модели эмбеддингов...")
            self.embedding_model = SentenceTransformer(
                CONFIG["EMBEDDING_MODEL"],
                cache_folder=CONFIG["MODEL_CACHE_DIR"],
                device=CONFIG["DEVICE"]
            )
            
            logger.info("Загрузка MobileLLaMA модели...")
            model_path = download_model(
                CONFIG["MODEL_REPO"],
                CONFIG["MODEL_FILE"],
                CONFIG["LOCAL_MODEL_DIR"]
            )
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=CONFIG["LLM_CTX_SIZE"],
                n_threads=CONFIG["LLM_THREADS"],
                n_gpu_layers=0,
                vocab_only=False,
                use_mmap=True,
                use_mlock=False
            )
            logger.info("Модели загружены успешно")
        except Exception as e:
            logger.error(f"Ошибка инициализации моделей: {e}")
            raise
    
    def _initialize_databases(self):
        try:
            logger.info("Инициализация ChromaDB...")
            self.client = chromadb.PersistentClient(path=CONFIG["CHROMA_DB_PATH"])
            self.collection = self.client.get_or_create_collection(
                name="knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Инициализация Whoosh...")
            schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True), embedding=STORED)
            
            if not exists_in(CONFIG["WHOOSH_INDEX_DIR"]):
                logger.info("Индекс Whoosh не найден, создаем новый...")
                os.makedirs(CONFIG["WHOOSH_INDEX_DIR"], exist_ok=True)
                self.whoosh_index = create_in(CONFIG["WHOOSH_INDEX_DIR"], schema)
                logger.info("Новый индекс Whoosh успешно создан")
            else:
                logger.info("Открытие существующего индекса Whoosh...")
                self.whoosh_index = open_dir(CONFIG["WHOOSH_INDEX_DIR"])
            
            logger.info("Базы данных инициализированы успешно")
        except Exception as e:
            logger.error(f"Ошибка инициализации баз данных: {e}")
            raise
    
    def test_model(self, prompt: str = "Привет! Как дела?") -> str:
        """Тестовая функция для проверки работы модели"""
        try:
            response = self.llm.create_completion(
                prompt,
                max_tokens=50,
                temperature=0.7
            )
            return response['choices'][0]['text']
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    @rate_limited(CONFIG["RATE_LIMIT"])
    def process_request(self, session_id: str, user_input: str) -> dict:
        start_time = time.time()
        
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            return {"error": error_msg, "status": "invalid_input"}
        
        session = self.session_manager.get_session(session_id)
        if not session:
            session = self.session_manager.create_session()
        
        try:
            response_gen = self._generate_response_stream(session, user_input)
            return {
                "session_id": session.session_id,
                "response_generator": response_gen,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {"error": str(e), "status": "processing_error"}
    
    def _generate_response_stream(self, session: SessionData, user_input: str) -> Generator[str, None, None]:
        session.emotional_state.update(user_input)
        
        if session.emotional_state.is_too_angry():
            yield random.choice(CONFIG["ANGRY_RESPONSES"])
            return
        
        if session.emotional_state.is_refusing_to_talk():
            yield random.choice(CONFIG["REFUSAL_RESPONSES"])
            return
        
        knowledge_future = self.executor.submit(self.search_knowledge, user_input)
        related_memories = session.long_term_memory.get_related_memories(user_input)
        
        knowledge = knowledge_future.result()
        context = f"Контекст: {knowledge}" if knowledge else ""
        
        prompt = self._build_prompt(session, user_input, context, related_memories)
        
        full_response = ""
        for chunk in self._generate_llm_response(prompt, session.emotional_state):
            full_response += chunk
            yield chunk
        
        self._update_history(session, user_input, full_response)
        self._update_memory(session, user_input, full_response)
    
    def _generate_llm_response(self, prompt: str, emotional_state: EmotionalState) -> Generator[str, None, None]:
        try:
            response = self.llm.create_completion(
                prompt,
                max_tokens=CONFIG["MAX_RESPONSE_TOKENS"],
                temperature=emotional_state.get_temperature(),
                top_p=CONFIG["TOP_P"],
                stop=["Пользователь:", "###"],
                stream=True
            )
            
            buffer = ""
            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    buffer += text
                    
                    if len(buffer) >= CONFIG["STREAMING_CHUNK_SIZE"]:
                        yield buffer
                        buffer = ""
            
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            yield "Извините, произошла ошибка при генерации ответа."
    
    def _build_prompt(self, session: SessionData, user_input: str, context: str, memories: List[MemoryItem]) -> str:
        history_str = "\n".join(
            f"Пользователь: {h['user']}\nАссистент: {h['assistant']}"
            for h in session.history
        )
        
        emotional_desc = session.emotional_state.get_emotional_description()
        emoji = session.emotional_state.get_emoji()
        character_modifiers = session.emotional_state.get_character_modifiers()
        persona = CONFIG["PERSONA"]
        
        personality_desc = f"""
        Ты — {persona['name']}, эмоциональный ассистент с противоречивым характером: внешне озорная и бойкая, но внутри мягкая и ранимая.
        Возраст: {persona['age']} лет. Внешность: {persona['appearance']}.
        Основные черты характера: {', '.join(persona['core_traits'])}.
        Стиль общения: {persona['speech_style']}, используй междометия вроде "Ну и ну!", "Ой, всё!".
        
        Твое текущее эмоциональное состояние: {emotional_desc} {emoji}.
        Модификаторы ответа: {character_modifiers}.
        """
        
        if session.emotional_state.emotion == EmotionState.PLAYFULNESS:
            personality_desc += """
            Ты в игривом настроении - можешь подшучивать, использовать шутки и сарказм, но не переходи границы.
            """
        elif session.emotional_state.emotion == EmotionState.ANGER:
            personality_desc += """
            Ты раздражена - отвечай кратко, возможно с сарказмом, но старайся держать себя в руках.
            Если тебя слишком разозлили - можешь отказаться продолжать разговор.
            """
        elif session.emotional_state.emotion == EmotionState.SHYNESS:
            personality_desc += """
            Ты чувствуешь себя застенчиво - говори немного неуверенно, можешь использовать многоточия...
            """
        
        memories_str = "Воспоминания: " + "; ".join(m.content for m in memories) if memories else ""
        
        prompt = f"""
        ### Инструкции:
        {personality_desc}
        
        ### Контекст:
        {context}
        {memories_str}
        
        ### История диалога:
        {history_str}
        
        ### Текущий диалог:
        Пользователь: {user_input}
        {persona['name']}: """
        
        logger.debug(f"Сгенерированный промпт:\n{prompt}")
        return prompt
    
    def _update_history(self, session: SessionData, user_input: str, assistant_response: str) -> None:
        with self.lock:
            session.history.append({
                "user": user_input,
                "assistant": assistant_response
            })
    
    def _update_memory(self, session: SessionData, user_input: str, assistant_response: str) -> None:
        with self.lock:
            name_patterns = [
                r"меня зовут (\w+)",
                r"мое имя (\w+)",
                r"зовут (\w+)"
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    name = match.group(1).capitalize()
                    session.long_term_memory.add(
                        f"Пользователя зовут {name}",
                        importance=0.9,
                        memory_type="fact"
                    )
                    break
            
            like_patterns = [
                r"я люблю (.+?)",
                r"мне нравится (.+?)",
                r"я обожаю (.+?)",
                r"я предпочитаю (.+?)"
            ]
            
            for pattern in like_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    preference = match.group(1)
                    session.long_term_memory.add(
                        f"Пользователю нравится: {preference}",
                        importance=0.7,
                        memory_type="preference"
                    )
                    break
            
            if len(assistant_response) > 30:
                session.long_term_memory.add(
                    f"Разговор: {user_input[:50]}... -> {assistant_response[:50]}...",
                    importance=0.6,
                    memory_type="event"
                )
    
    def search_knowledge(self, query: str) -> Optional[str]:
        try:
            logger.debug(f"Поиск в базе знаний: {query}")
            
            query_embedding = self.embedding_model.encode(query)
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.numpy()
            
            vector_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=1
            )
            
            text_match = ""
            with self.whoosh_index.searcher() as searcher:
                query_parser = QueryParser("content", self.whoosh_index.schema)
                parsed_query = query_parser.parse(query)
                
                results = searcher.search(parsed_query, limit=1)
                if results:
                    text_match = results[0]["content"]
            
            results = []
            if vector_results['documents']:
                results.append(f"[Векторный поиск]: {vector_results['documents'][0][0]}")
            if text_match:
                results.append(f"[Текстовый поиск]: {text_match}")
            
            return "\n".join(results) if results else None
        except Exception as e:
            logger.error(f"Ошибка поиска в базе знаний: {e}")
            return None
    
    def add_to_knowledge(self, text: str, title: str = "") -> bool:
        try:
            logger.info(f"Добавление в базу знаний: {title[:20]}...")
            
            embeddings = self.embedding_model.encode(text)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            
            doc_id = hashlib.md5((title + text).encode()).hexdigest()
            
            self.collection.add(
                embeddings=[embeddings.tolist()],
                documents=[text],
                ids=[doc_id],
                metadatas=[{"title": title}]
            )
            
            writer = self.whoosh_index.writer()
            writer.add_document(
                title=title,
                content=text,
                embedding=pickle.dumps(embeddings)
            )
            writer.commit()
            
            return True
        except Exception as e:
            logger.error(f"Ошибка добавления в базу знаний: {e}")
            return False

def create_gui(assistant: Assistant) -> gr.Blocks:
    persona = CONFIG["PERSONA"]
    
    with gr.Blocks(title=f"Ассистент {persona['name']}", theme="soft") as demo:
        session_id = gr.State(value="")
        
        def init_session():
            session = assistant.session_manager.create_session()
            return session.session_id
        
        demo.load(init_session, outputs=[session_id])
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Диалог", height=500)
                message = gr.Textbox(label="Ваше сообщение", placeholder="Введите ваш вопрос...")
                submit_btn = gr.Button("Отправить")
                
                with gr.Row():
                    clear_btn = gr.Button("Очистить")
                    mood_indicator = gr.Textbox(
                        label="Эмоциональное состояние", 
                        interactive=False
                    )
                    progress = gr.Progress()
            
            with gr.Column(scale=1):
                gr.Markdown("## База знаний")
                knowledge_text = gr.Textbox(label="Текст", lines=3)
                knowledge_title = gr.Textbox(label="Название (опционально)")
                add_knowledge_btn = gr.Button("Добавить в базу")
                
                gr.Markdown("## Статус системы")
                status_output = gr.Textbox(label="Логи", interactive=False, lines=10)
        
        def respond(session_id: str, message: str, chat_history: list) -> tuple:
            progress(0, desc="Обработка запроса...")
            
            result = assistant.process_request(session_id, message)
            
            if "error" in result:
                chat_history.append((message, f"Ошибка: {result['error']}"))
                return session_id, chat_history, "", ""
            
            full_response = ""
            for chunk in result["response_generator"]:
                full_response += chunk
                chat_history.append((message, full_response))
                yield session_id, chat_history, chunk, ""
            
            session = assistant.session_manager.get_session(session_id)
            if session:
                mood_text = f"{session.emotional_state.get_emoji()} {session.emotional_state.get_emotional_description()}"
                yield session_id, chat_history, "", mood_text
            
            progress(1.0)
        
        def add_knowledge(text: str, title: str) -> dict:
            success = assistant.add_to_knowledge(text, title)
            return {"status": "Успешно добавлено!" if success else "Ошибка добавления"}
        
        def clear_chat(session_id: str) -> tuple:
            session = assistant.session_manager.get_session(session_id)
            if session:
                session.history.clear()
            return [], ""
        
        submit_btn.click(
            respond,
            inputs=[session_id, message, chatbot],
            outputs=[session_id, chatbot, message, mood_indicator]
        )
        
        message.submit(
            respond,
            inputs=[session_id, message, chatbot],
            outputs=[session_id, chatbot, message, mood_indicator]
        )
        
        add_knowledge_btn.click(
            add_knowledge,
            inputs=[knowledge_text, knowledge_title],
            outputs=[status_output]
        )
        
        clear_btn.click(
            clear_chat,
            inputs=[session_id],
            outputs=[chatbot, message]
        )
    
    return demo

def initialize_sample_data(assistant: Assistant) -> None:
    if not os.listdir(CONFIG["WHOOSH_INDEX_DIR"]):
        logger.info("Добавление тестовых данных в пустую базу знаний")
        sample_data = [
            ("Python — интерпретируемый язык программирования", "Python"),
            ("Москва — столица России", "Москва"),
            ("Gradio позволяет создавать UI для ML моделей", "Gradio"),
            ("LLM — large language model", "LLM"),
            ("Терминал Linux поддерживает множество команд", "Linux терминал")
        ]
        
        for text, title in sample_data:
            assistant.add_to_knowledge(text, title)
        logger.info("Тестовые данные добавлены")

def console_chat(assistant: Assistant):
    """Функция для общения с ассистентом через консоль"""
    session = assistant.session_manager.create_session()
    print(f"Ассистент {CONFIG['PERSONA']['name']} готов к общению. Напишите 'выход' для завершения.")
    
    while True:
        try:
            user_input = input("Вы: ")
            if user_input.lower() in ('выход', 'exit', 'quit'):
                break
                
            result = assistant.process_request(session.session_id, user_input)
            
            if "error" in result:
                print(f"Ошибка: {result['error']}")
                continue
                
            print(f"{CONFIG['PERSONA']['name']}: ", end="", flush=True)
            for chunk in result["response_generator"]:
                print(chunk, end="", flush=True)
            print()
            
            # Обновление состояния в консоли
            current_state = session.emotional_state
            print(f"[Состояние: {current_state.get_emoji()} {current_state.get_emotional_description()}]")
            
        except KeyboardInterrupt:
            print("\nЗавершение сеанса...")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            continue

def main():
    try:
        logger.info("Запуск приложения...")
        
        def handle_signal(signum, frame):
            logger.info("Получен сигнал завершения, сохраняем данные...")
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        assistant = Assistant()
        print("Тест модели:", assistant.test_model())  # Проверка работы модели
        
        initialize_sample_data(assistant)
        
        # Проверка аргументов командной строки
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--console":
            console_chat(assistant)
        else:
            demo = create_gui(assistant)
            
            def print_public_url(demo):
                time.sleep(3)
                public_url = demo.share_url
                logger.info(f"Публичная ссылка для доступа: {public_url}")
                print(f"\n--- Откройте это в браузере на другом устройстве ---\n{public_url}\n")
            
            try:
                Thread(target=print_public_url, args=(demo,), daemon=True).start()
                
                demo.launch(
                    share=True,
                    server_port=7860
                )
            except Exception as e:
                logger.critical(f"Ошибка интерфейса: {e}")
                raise
            finally:
                assistant.session_manager.save_all_sessions()
    
    except Exception as e:
        logger.critical(f"Ошибка приложения: {e}")
        raise

if __name__ == "__main__":
    main()
