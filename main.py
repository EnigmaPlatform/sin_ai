
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import uuid
import zipfile
import shutil

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

def check_and_move_model():
    """Проверяет наличие модели в Download и перемещает в нужную директорию"""
    # Исходный путь к модели
    source_path = "/storage/emulated/0/Download/mobilellama-1.4b-chat-q4_k_m-imat.gguf"
    # Целевой путь
    target_dir = CONFIG["LOCAL_MODEL_DIR"]
    target_path = os.path.join(target_dir, CONFIG["MODEL_FILE"])
    # Проверяем, существует ли модель в целевой директории
    if os.path.exists(target_path):
        logger.info(f"Модель уже находится в целевой директории: {target_path}")
        return target_path
    # Проверяем, есть ли модель в исходной директории
    if not os.path.exists(source_path):
        logger.info(f"Модель не найдена в директории Download: {source_path}")
        return None
    try:
        # Создаем целевую директорию, если ее нет
        os.makedirs(target_dir, exist_ok=True)
        # Перемещаем файл
        shutil.move(source_path, target_path)
        logger.info(f"Модель успешно перемещена из {source_path} в {target_path}")
        # Проверяем, что файл перемещен корректно
        if os.path.exists(target_path):
            logger.info("Проверка перемещения успешна")
            return target_path
        else:
            logger.error("Ошибка: файл не появился в целевой директории после перемещения")
            return None
    except Exception as e:
        logger.error(f"Ошибка при перемещении модели: {str(e)}")
        return None

# --- Расширенная конфигурация ---
CONFIG = {
    "MODEL_REPO": "marroyo777/MobileLLaMA-1.4B-Chat-Q4_K_M-GGUF",
    "MODEL_FILE": "mobilellama-1.4b-chat-q4_k_m-imat.gguf",
    "LOCAL_MODEL_DIR": "/storage/emulated/0/Download/models",
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Исправлено: модель paraphrase-multilingual-MiniLM-L12-v2 генерирует эмбеддинги размерности 384
    "EMBEDDING_DIM": 384, 
    "MAX_HISTORY": 10,
    "WHOOSH_INDEX_DIR": "whoosh_index",
    "CHROMA_DB_PATH": "chroma_db",
    "MODEL_CACHE_DIR": "/storage/emulated/0/Download/models_cache",
    "DEVICE": "cpu",
    "LOCAL_FILES_ONLY": False,
    "HF_HOME": "/storage/emulated/0/Download/models_cache",
    "LLM_CTX_SIZE": 2048,
    "LLM_THREADS": 4,
    "MOOD_UPDATE_INTERVAL": 60,
    "MOOD_DECAY_RATE": 0.95,
    "MOOD_VOLATILITY": 0.1,
    "MEMORY_DECAY_RATE": 0.98,
    "LONG_TERM_MEMORY_SIZE": 500,
    "MAX_RESPONSE_TOKENS": 150,
    "TOP_P": 0.9,
    "TEMPERATURE_RANGE": (0.5, 1.0),
    "MEMORY_IMPORTANCE_THRESHOLD": 0.3,
    "MAX_MEMORY_RETRIEVAL": 5,
    "RATE_LIMIT": 3,
    "MAX_SESSION_AGE": 86400,
    "STREAMING_CHUNK_SIZE": 30,
    "MAX_INPUT_LENGTH": 1000,
    "COMPRESS_MEMORY": True,
    "THREAD_POOL_SIZE": 4,
    "USER_PROFILES_DIR": "user_profiles",
    "PLANNING_MAX_STEPS": 5,
    "PLANNING_TEMP": 0.7,
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
for dir_key in ["LOCAL_MODEL_DIR", "WHOOSH_INDEX_DIR", "CHROMA_DB_PATH", "MODEL_CACHE_DIR", "USER_PROFILES_DIR"]:
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
    # Сначала проверяем, не перемещена ли уже модель
    model_path = check_and_move_model()
    if model_path:
        return model_path
    # Если модель не найдена нигде, загружаем стандартным способом
    model_path = os.path.join(local_dir, filename)
    # Проверяем, существует ли модель локально
    if os.path.exists(model_path):
        logger.info(f"Модель {filename} уже существует локально по пути: {model_path}")
        return model_path
    logger.info(f"Начинаем загрузку модели {filename}...")
    try:
        # Проверяем, есть ли модель в кэше huggingface
        cache_path = os.path.join(CONFIG["HF_HOME"], f"models--{repo_id.replace('/', '--')}")
        if os.path.exists(cache_path):
            logger.info(f"Найдена модель в кэше huggingface: {cache_path}")
            # Ищем файл модели в кэше
            for root, _, files in os.walk(cache_path):
                if filename in files:
                    cached_model_path = os.path.join(root, filename)
                    # Копируем в целевую директорию
                    os.makedirs(local_dir, exist_ok=True)
                    shutil.copy2(cached_model_path, model_path)
                    logger.info(f"Модель скопирована из кэша в {model_path}")
                    return model_path
        # Если модель не найдена ни локально, ни в кэше, загружаем
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
        self.associated_emotions = []
        self.context_tags = []

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

    def add_emotion_context(self, emotion: EmotionState, intensity: float):
        with self.lock:
            self.associated_emotions.append({
                "emotion": emotion,
                "intensity": intensity,
                "timestamp": time.time()
            })

    def add_context_tag(self, tag: str):
        with self.lock:
            if tag not in self.context_tags:
                self.context_tags.append(tag)

    def __str__(self):
        return f"{self.content} (важность: {self.importance:.2f}, тип: {self.memory_type})"

class LongTermMemory:
    def __init__(self, max_size: int = 100):
        self.memories = []
        self.max_size = max_size
        self.lock = Lock()
        self.embedding_model = None
        self._initialize_embedding_model()
        self.memory_clusters = []
        self.cluster_last_updated = 0
        self.memory_graph = defaultdict(list)  # Граф связей между воспоминаниями

    def _initialize_embedding_model(self):
        try:
            # Проверяем, есть ли модель эмбеддингов локально
            embedding_model_path = os.path.join(
                CONFIG["MODEL_CACHE_DIR"],
                "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"
            )
            if os.path.exists(embedding_model_path):
                required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                model_files = os.listdir(embedding_model_path)
                if all(file in model_files for file in required_files):
                    logger.info(f"Модель эмбеддингов уже существует в {embedding_model_path}, загружаем...")
                    self.embedding_model = SentenceTransformer(
                        embedding_model_path,
                        device=CONFIG["DEVICE"]
                    )
                    return
            # Если модель не найдена локально, загружаем стандартным способом
            self.embedding_model = SentenceTransformer(
                CONFIG["EMBEDDING_MODEL"],
                cache_folder=CONFIG["MODEL_CACHE_DIR"],
                device=CONFIG["DEVICE"]
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации модели эмбеддингов для памяти: {e}")
            raise

    def add(self, content: str, importance: float = 0.5, memory_type: str = "fact", 
            emotion_context: Optional[EmotionState] = None, emotion_intensity: float = 0.0,
            context_tags: List[str] = None):
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
            if emotion_context:
                new_memory.add_emotion_context(emotion_context, emotion_intensity)
            if context_tags:
                for tag in context_tags:
                    new_memory.add_context_tag(tag)
            self.memories.append(new_memory)
            self._update_memory_connections(new_memory)
            # Периодически обновляем кластеры
            if time.time() - self.cluster_last_updated > 3600:  # Каждый час
                self._cluster_memories()
                self.cluster_last_updated = time.time()

    def _update_memory_connections(self, new_memory: MemoryItem):
        """Обновляет связи между воспоминаниями на основе семантической близости"""
        if not new_memory.embedding:
            return
        for mem in self.memories:
            if mem == new_memory or not mem.embedding:
                continue
            similarity = cosine_similarity(
                [new_memory.embedding],
                [mem.embedding]
            )[0][0]
            if similarity > 0.7:  # Порог для создания связи
                self.memory_graph[new_memory.content].append((mem.content, similarity))
                self.memory_graph[mem.content].append((new_memory.content, similarity))

    def _cluster_memories(self):
        """Кластеризует воспоминания для лучшей организации"""
        if not self.memories or len(self.memories) < 3:
            return
        try:
            embeddings = [m.embedding for m in self.memories if m.embedding is not None]
            if not embeddings:
                return
            # Уменьшаем размерность для кластеризации
            pca = PCA(n_components=min(10, len(embeddings[0])))
            reduced_embeddings = pca.fit_transform(embeddings)
            # Определяем оптимальное количество кластеров
            max_clusters = min(10, len(reduced_embeddings))
            distortions = []
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(reduced_embeddings)
                distortions.append(kmeans.inertia_)
            # Метод локтя для выбора k
            optimal_k = 3
            if len(distortions) > 3:
                deltas = [distortions[i] - distortions[i+1] for i in range(len(distortions)-1)]
                optimal_k = deltas.index(max(deltas)) + 1
            # Финальная кластеризация
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(reduced_embeddings)
            # Обновляем кластеры
            self.memory_clusters = []
            idx = 0
            for mem in self.memories:
                if mem.embedding is not None:
                    self.memory_clusters.append((mem.content, clusters[idx]))
                    idx += 1
                else:
                    self.memory_clusters.append((mem.content, -1))
            logger.info(f"Обновлены кластеры памяти: {optimal_k} кластеров")
        except Exception as e:
            logger.error(f"Ошибка кластеризации памяти: {e}")

    def retrieve(self, query: str = None, n: int = 3, emotion_filter: Optional[EmotionState] = None,
                min_importance: float = 0.0, context_tags: List[str] = None) -> List[MemoryItem]:
        with self.lock:
            for memory in self.memories:
                memory.decay()
            if not query:
                # Возвращаем самые важные воспоминания с фильтрами
                filtered = self._filter_memories(emotion_filter, min_importance, context_tags)
                sorted_memories = sorted(filtered, key=lambda x: -x.importance)
                return sorted_memories[:n]
            try:
                query_embedding = self.embedding_model.encode(query)
                if isinstance(query_embedding, torch.Tensor):
                    query_embedding = query_embedding.numpy()
                similarities = []
                for memory in self._filter_memories(emotion_filter, min_importance, context_tags):
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
                filtered = self._filter_memories(emotion_filter, min_importance, context_tags)
                return sorted(filtered, key=lambda x: -x.importance)[:n]

    def _filter_memories(self, emotion_filter: Optional[EmotionState] = None,
                        min_importance: float = 0.0, context_tags: List[str] = None) -> List[MemoryItem]:
        """Фильтрует воспоминания по заданным критериям"""
        filtered = []
        for mem in self.memories:
            if mem.importance < min_importance:
                continue
            if emotion_filter:
                has_emotion = any(e["emotion"] == emotion_filter for e in mem.associated_emotions)
                if not has_emotion:
                    continue
            if context_tags:
                has_all_tags = all(tag in mem.context_tags for tag in context_tags)
                if not has_all_tags:
                    continue
            filtered.append(mem)
        return filtered

    def get_related_memories(self, topic: str, n: int = 3) -> List[MemoryItem]:
        with self.lock:
            related = self.retrieve(topic, n)
            for memory in related:
                memory.access()
            # Добавляем связанные воспоминания из графа
            if topic in self.memory_graph:
                related_contents = [r.content for r in related]
                connections = sorted(self.memory_graph[topic], key=lambda x: -x[1])[:n]
                for content, _ in connections:
                    if content not in related_contents:
                        mem = next((m for m in self.memories if m.content == content), None)
                        if mem:
                            related.append(mem)
                            if len(related) >= n*2:
                                break
            return related[:n*2]  # Возвращаем больше результатов, если есть связи

    def get_cluster_memories(self, cluster_id: int) -> List[MemoryItem]:
        """Возвращает воспоминания из определенного кластера"""
        with self.lock:
            if not self.memory_clusters or cluster_id < 0:
                return []
            cluster_contents = [content for content, cid in self.memory_clusters if cid == cluster_id]
            return [mem for mem in self.memories if mem.content in cluster_contents]

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
                        mem_copy.associated_emotions = mem.associated_emotions.copy()
                        mem_copy.context_tags = mem.context_tags.copy()
                        memories_to_save.append(mem_copy)
                    data = {
                        'memories': memories_to_save,
                        'memory_graph': dict(self.memory_graph),
                        'memory_clusters': self.memory_clusters.copy(),
                        'cluster_last_updated': self.cluster_last_updated
                    }
                    pickle.dump(data, f)
            except Exception as e:
                logger.error(f"Ошибка сохранения памяти: {e}")
                raise

    def load(self, filepath: str):
        with self.lock:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        self.memories = data.get('memories', [])
                        self.memory_graph = defaultdict(list, data.get('memory_graph', {}))
                        self.memory_clusters = data.get('memory_clusters', [])
                        self.cluster_last_updated = data.get('cluster_last_updated', 0)
                        # Восстанавливаем эмбеддинги
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
                    self.memory_graph = defaultdict(list)
                    self.memory_clusters = []
                    self.cluster_last_updated = 0

class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.profile_file = os.path.join(CONFIG["USER_PROFILES_DIR"], f"{user_id}.json")
        self.data = {
            "basic_info": {},
            "preferences": {},
            "habits": {},
            "relationships": {},
            "conversation_history": [],
            "custom_facts": [],
            "learning_style": "balanced",
            "last_updated": time.time()
        }
        self.lock = Lock()
        self.load()

    def load(self):
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки профиля пользователя {self.user_id}: {e}")

    def save(self):
        with self.lock:
            self.data["last_updated"] = time.time()
            try:
                with open(self.profile_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Ошибка сохранения профиля пользователя {self.user_id}: {e}")

    def update_basic_info(self, info: dict):
        with self.lock:
            self.data["basic_info"].update(info)
            self.save()

    def update_preference(self, category: str, item: str, value: Any):
        with self.lock:
            if category not in self.data["preferences"]:
                self.data["preferences"][category] = {}
            self.data["preferences"][category][item] = value
            self.save()

    def add_conversation_event(self, event_type: str, data: dict):
        with self.lock:
            self.data["conversation_history"].append({
                "type": event_type,
                "data": data,
                "timestamp": time.time()
            })
            # Сохраняем только последние 100 событий
            if len(self.data["conversation_history"]) > 100:
                self.data["conversation_history"] = self.data["conversation_history"][-100:]
            self.save()

    def add_custom_fact(self, fact: str, importance: float = 0.5):
        with self.lock:
            self.data["custom_facts"].append({
                "fact": fact,
                "importance": importance,
                "timestamp": time.time()
            })
            self.save()

    def get_preferred_topics(self, n: int = 5) -> List[str]:
        with self.lock:
            if not self.data["preferences"].get("topics", {}):
                return []
            sorted_topics = sorted(
                self.data["preferences"]["topics"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [topic for topic, score in sorted_topics[:n]]

    def get_disliked_topics(self, n: int = 3) -> List[str]:
        with self.lock:
            if not self.data["preferences"].get("topics", {}):
                return []
            sorted_topics = sorted(
                self.data["preferences"]["topics"].items(),
                key=lambda x: x[1]
            )
            return [topic for topic, score in sorted_topics[:n]]

    def get_learning_style(self) -> str:
        with self.lock:
            return self.data.get("learning_style", "balanced")

    def update_learning_style(self, style: str):
        with self.lock:
            if style in ["visual", "auditory", "kinesthetic", "balanced"]:
                self.data["learning_style"] = style
                self.save()

class Planner:
    def __init__(self, llm):
        self.llm = llm
        self.plans = {}
        self.lock = Lock()

    def create_plan(self, goal: str, session_id: str) -> List[str]:
        """Создает план действий для достижения цели"""
        with self.lock:
            if session_id in self.plans:
                return self.plans[session_id]
            prompt = f"""
            Ты - помощник в планировании. Разбей цель на последовательные шаги.
            Цель: {goal}
            Шаги (не более {CONFIG["PLANNING_MAX_STEPS"]}):
            1."""
            try:
                response = self.llm.create_completion(
                    prompt,
                    max_tokens=300,
                    temperature=CONFIG["PLANNING_TEMP"],
                    stop=["###", "Цель:"]
                )
                steps_text = response['choices'][0]['text'].strip()
                steps = ["1. " + step.strip() for step in steps_text.split('\n') if step.strip()]
                # Ограничиваем количество шагов
                steps = steps[:CONFIG["PLANNING_MAX_STEPS"]]
                self.plans[session_id] = steps
                logger.info(f"Создан план для цели '{goal}': {steps}")
                return steps
            except Exception as e:
                logger.error(f"Ошибка создания плана: {e}")
                return ["Не удалось создать план действий"]

    def execute_step(self, session_id: str, step_index: int, context: str) -> str:
        """Выполняет конкретный шаг плана"""
        with self.lock:
            if session_id not in self.plans or step_index >= len(self.plans[session_id]):
                return "Неверный шаг плана"
            step = self.plans[session_id][step_index]
            prompt = f"""
            Ты выполняешь шаг из плана. Вот контекст:
            {context}
            Шаг, который нужно выполнить: {step}
            Действие:"""
            try:
                response = self.llm.create_completion(
                    prompt,
                    max_tokens=200,
                    temperature=CONFIG["PLANNING_TEMP"],
                    stop=["###", "Шаг"]
                )
                action = response['choices'][0]['text'].strip()
                logger.info(f"Выполнен шаг {step_index}: {action}")
                return action
            except Exception as e:
                logger.error(f"Ошибка выполнения шага плана: {e}")
                return f"Не удалось выполнить шаг: {step}"

    def update_plan(self, session_id: str, new_steps: List[str]):
        """Обновляет план для сессии"""
        with self.lock:
            self.plans[session_id] = new_steps

    def clear_plan(self, session_id: str):
        """Очищает план для сессии"""
        with self.lock:
            if session_id in self.plans:
                del self.plans[session_id]

@dataclass
class SessionData:
    session_id: str
    user_id: str
    history: deque
    emotional_state: EmotionalState
    long_term_memory: LongTermMemory
    created_at: float
    last_accessed: float
    planner: Planner
    current_plan: Optional[List[str]] = None
    current_step: int = 0

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG["THREAD_POOL_SIZE"])
        self.user_profiles: Dict[str, UserProfile] = {}

    def create_session(self, user_id: str = None) -> SessionData:
        session_id = str(uuid.uuid4())
        if not user_id:
            user_id = f"anonymous_{hashlib.md5(session_id.encode()).hexdigest()[:8]}"
        # Инициализация профиля пользователя, если еще не существует
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        # Создание планировщика
        planner = Planner(None)  # Инициализируется позже, когда будет доступна модель LLM
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            history=deque(maxlen=CONFIG["MAX_HISTORY"]),
            emotional_state=EmotionalState(),
            long_term_memory=LongTermMemory(CONFIG["LONG_TERM_MEMORY_SIZE"]),
            created_at=time.time(),
            last_accessed=time.time(),
            planner=planner
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

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        with self.lock:
            return self.user_profiles.get(user_id)

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
            # Сохраняем историю диалога в профиль пользователя
            if session.user_id in self.user_profiles:
                for item in session.history:
                    self.user_profiles[session.user_id].add_conversation_event(
                        "dialog",
                        {"user": item["user"], "assistant": item["assistant"]}
                    )
        except Exception as e:
            logger.error(f"Ошибка сохранения данных сессии {session.session_id}: {e}")

    def save_all_sessions(self):
        with self.lock:
            for session in self.sessions.values():
                self._save_session_data(session)
            # Сохраняем все профили пользователей
            for profile in self.user_profiles.values():
                profile.save()

    def __del__(self):
        self.save_all_sessions()
        self.executor.shutdown(wait=True)

class Assistant:
    def __init__(self):
        logger.info("Инициализация ассистента...")
        self.session_manager = SessionManager()
        self._initialize_models()
        self._initialize_databases()
        self.rate_limit_cache = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG["THREAD_POOL_SIZE"])
        self._start_cleanup_task()
        logger.info("Ассистент успешно инициализирован")

    def _start_cleanup_task(self):
        def cleanup():
            while True:
                time.sleep(3600)
                self.session_manager.cleanup_sessions()
        Thread(target=cleanup, daemon=True).start()

    def _initialize_models(self):
        try:
            logger.info("Загрузка облегченной модели эмбеддингов...")
            # Проверяем, есть ли модель эмбеддингов локально
            embedding_model_path = os.path.join(
                CONFIG["MODEL_CACHE_DIR"],
                "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"
            )
            if os.path.exists(embedding_model_path):
                required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                model_files = os.listdir(embedding_model_path)
                if all(file in model_files for file in required_files):
                    logger.info(f"Модель эмбеддингов уже существует в {embedding_model_path}, загружаем...")
                    self.embedding_model = SentenceTransformer(
                        embedding_model_path,
                        device=CONFIG["DEVICE"]
                    )
                else:
                    logger.info("Не все файлы модели найдены, загружаем заново...")
                    self.embedding_model = SentenceTransformer(
                        CONFIG["EMBEDDING_MODEL"],
                        cache_folder=CONFIG["MODEL_CACHE_DIR"],
                        device=CONFIG["DEVICE"]
                    )
            else:
                logger.info("Модель эмбеддингов не найдена, загружаем...")
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
            # Инициализируем планировщик с моделью LLM
            for session in self.session_manager.sessions.values():
                session.planner = Planner(self.llm)
            logger.info("Модели загружены успешно")
        except Exception as e:
            logger.error(f"Ошибка инициализации моделей: {e}")
            raise

    def _initialize_databases(self):
        try:
            logger.info("Инициализация ChromaDB...")
            self.client = chromadb.PersistentClient(path=CONFIG["CHROMA_DB_PATH"])
            # Убедимся, что размерность в коллекции соответствует размерности модели
            self.collection = self.client.get_or_create_collection(
                name="knowledge",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None # Используем внешнюю функцию
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
            return {"error": f"Произошла внутренняя ошибка при обработке запроса: {str(e)}", "status": "processing_error"}

    def _generate_response_stream(self, session: SessionData, user_input: str) -> Generator[str, None, None]:
        try:
            session.emotional_state.update(user_input)
            if session.emotional_state.is_too_angry():
                yield random.choice(CONFIG["ANGRY_RESPONSES"])
                return
            if session.emotional_state.is_refusing_to_talk():
                yield random.choice(CONFIG["REFUSAL_RESPONSES"])
                return
            # Проверяем, не является ли ввод командой планирования
            planning_response = self._handle_planning_commands(session, user_input)
            if planning_response:
                yield planning_response
                return
            # Получаем знания и воспоминания параллельно
            knowledge_future = self.executor.submit(self.search_knowledge, user_input)
            related_memories = session.long_term_memory.get_related_memories(user_input)
            # Получаем профиль пользователя для персонализации
            user_profile = self.session_manager.get_user_profile(session.user_id)
            user_context = self._get_user_context(user_profile) if user_profile else ""
            knowledge = knowledge_future.result()
            context = f"Контекст: {knowledge}" if knowledge else ""
            prompt = self._build_prompt(session, user_input, context, related_memories, user_context)
            
            full_response = ""
            try:
                for chunk in self._generate_llm_response(prompt, session.emotional_state):
                    full_response += chunk
                    yield chunk
            except Exception as e:
                 logger.error(f"Ошибка генерации ответа LLM: {e}")
                 yield "Извините, произошла ошибка при генерации ответа."
                 return # Завершаем генерацию после ошибки

            self._update_history(session, user_input, full_response)
            self._update_memory(session, user_input, full_response)
            self._update_user_profile(session, user_input, full_response)
        except Exception as e:
            logger.error(f"Ошибка в _generate_response_stream: {e}")
            yield "Извините, произошла внутренняя ошибка при обработке вашего запроса."

    def _handle_planning_commands(self, session: SessionData, user_input: str) -> Optional[str]:
        """Обрабатывает команды, связанные с планированием"""
        planning_triggers = [
            "составь план",
            "как сделать",
            "пошагово",
            "алгоритм действий",
            "что нужно сделать"
        ]
        if any(trigger in user_input.lower() for trigger in planning_triggers):
            # Создаем новый план
            session.current_plan = session.planner.create_plan(user_input, session.session_id)
            session.current_step = 0
            plan_text = "\n".join(session.current_plan)
            return f"Я составила план:\n{plan_text}\nНачнем с первого шага?"
        if "следующий шаг" in user_input.lower() and session.current_plan:
            # Выполняем следующий шаг плана
            if session.current_step < len(session.current_plan):
                context = f"История: {list(session.history)[-2:]}" if len(session.history) >= 2 else ""
                action = session.planner.execute_step(
                    session.session_id,
                    session.current_step,
                    context
                )
                session.current_step += 1
                return action
            else:
                session.current_plan = None
                session.current_step = 0
                return "План выполнен! Что будем делать дальше?"
        return None

    def _get_user_context(self, user_profile: UserProfile) -> str:
        """Генерирует контекст на основе профиля пользователя"""
        context_parts = []
        # Базовая информация
        if user_profile.data["basic_info"]:
            basic_info = ", ".join(f"{k}: {v}" for k, v in user_profile.data["basic_info"].items())
            context_parts.append(f"Пользователь: {basic_info}")
        # Предпочтения
        preferred_topics = user_profile.get_preferred_topics()
        disliked_topics = user_profile.get_disliked_topics()
        if preferred_topics:
            context_parts.append(f"Любимые темы: {', '.join(preferred_topics)}")
        if disliked_topics:
            context_parts.append(f"Нелюбимые темы: {', '.join(disliked_topics)}")
        # Последние факты
        if user_profile.data["custom_facts"]:
            last_facts = user_profile.data["custom_facts"][-3:]  # Последние 3 факта
            facts_text = "; ".join(f["fact"] for f in last_facts)
            context_parts.append(f"Последние факты: {facts_text}")
        return "\n".join(context_parts) if context_parts else ""

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
             # Логируем ошибку, но не выкидываем её, чтобы поток не прерывался
             logger.error(f"Ошибка в _generate_llm_response: {e}") 
             # Можно отправить сообщение об ошибке клиенту
             yield "Извините, произошла ошибка при генерации ответа."
             # Здесь важно не вызывать raise, чтобы функция завершилась нормально
             # и генератор мог быть корректно обработан вызывающим кодом

    def _build_prompt(self, session: SessionData, user_input: str, context: str, 
                     memories: List[MemoryItem], user_context: str = "") -> str:
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
        
        # Добавляем проверки на None и пустоту
        context_str = context if context else ""
        memories_str = "Воспоминания: " + "; ".join(m.content for m in memories) if memories else ""
        
        prompt = f"""
        ### Инструкции:
        {personality_desc}
        ### Контекст пользователя:
        {user_context}
        ### Общий контекст:
        {context_str}
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
            # Извлекаем и запоминаем имена
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
                        memory_type="fact",
                        emotion_context=session.emotional_state.emotion,
                        emotion_intensity=session.emotional_state.emotion_intensity,
                        context_tags=["имя", "личное"]
                    )
                    # Обновляем профиль пользователя
                    if session.user_id.startswith("anonymous_"):
                        new_user_id = f"user_{name.lower()}"
                        session.user_id = new_user_id
                    break
            # Извлекаем предпочтения
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
                        memory_type="preference",
                        emotion_context=session.emotional_state.emotion,
                        emotion_intensity=session.emotional_state.emotion_intensity,
                        context_tags=["предпочтение"]
                    )
                    # Обновляем профиль пользователя
                    if session.user_id in self.session_manager.user_profiles:
                        self.session_manager.user_profiles[session.user_id].update_preference(
                            "topics", preference, 1.0
                        )
                    break
            # Запоминаем важные моменты диалога
            if len(assistant_response) > 30:
                session.long_term_memory.add(
                    f"Разговор: {user_input[:50]}... -> {assistant_response[:50]}...",
                    importance=0.6,
                    memory_type="event",
                    emotion_context=session.emotional_state.emotion,
                    emotion_intensity=session.emotional_state.emotion_intensity,
                    context_tags=["диалог"]
                )

    def _update_user_profile(self, session: SessionData, user_input: str, assistant_response: str) -> None:
        """Обновляет профиль пользователя на основе диалога"""
        if session.user_id not in self.session_manager.user_profiles:
            return
        profile = self.session_manager.user_profiles[session.user_id]
        # Анализируем пользовательский ввод на предмет личной информации
        personal_info_patterns = {
            "age": r"мне (\d+) лет",
            "city": r"я из (\w+)",
            "job": r"я работаю (\w+)",
            "hobby": r"мое хобби (\w+)"
        }
        for field, pattern in personal_info_patterns.items():
            match = re.search(pattern, user_input.lower())
            if match:
                profile.update_basic_info({field: match.group(1)})
        # Анализируем эмоциональные реакции пользователя
        emotion_triggers = CONFIG["EMOTION_TRIGGERS"]
        for category, triggers in emotion_triggers.items():
            if any(word in user_input.lower() for word in triggers["words"]):
                profile.add_conversation_event(
                    "emotion_reaction",
                    {
                        "category": category,
                        "text": user_input,
                        "assistant_response": assistant_response
                    }
                )
                break

    def search_knowledge(self, query: str) -> Optional[str]:
        try:
            logger.debug(f"Поиск в базе знаний: {query}")
            query_embedding = self.embedding_model.encode(query)
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.numpy()

            # Исправлено: Проверка на пустой результат из ChromaDB
            vector_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=1
            )
            
            # Проверяем, есть ли результаты из ChromaDB
            chroma_result = ""
            if vector_results['documents'] and vector_results['documents'][0]:
                 chroma_result = f"[Векторный поиск]: {vector_results['documents'][0][0]}"

            text_match = ""
            with self.whoosh_index.searcher() as searcher:
                query_parser = QueryParser("content", self.whoosh_index.schema)
                parsed_query = query_parser.parse(query)
                results = searcher.search(parsed_query, limit=1)
                if results:
                    text_match = results[0]["content"]

            results = []
            if chroma_result: # Используем исправленный результат
                results.append(chroma_result)
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
                with gr.Row():
                    submit_btn = gr.Button("Отправить")
                    clear_btn = gr.Button("Очистить")
                with gr.Row():
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
            assistant_response = ""
            try:
                for chunk in result["response_generator"]:
                    assistant_response += chunk
                    # Обновляем историю чата по мере получения чанков
                    yield session_id, chat_history + [(message, assistant_response)], "", ""
            except Exception as e:
                 logger.error(f"Ошибка при получении потока ответа: {e}")
                 chat_history.append((message, "Ошибка: Не удалось получить ответ."))
                 yield session_id, chat_history, "", ""
                 return # Завершаем функцию после ошибки

            session = assistant.session_manager.get_session(session_id)
            if session:
                mood_text = f"{session.emotional_state.get_emoji()} {session.emotional_state.get_emotional_description()}"
                yield session_id, chat_history + [(message, assistant_response)], "", mood_text
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
            try:
                for chunk in result["response_generator"]:
                    print(chunk, end="", flush=True)
            except Exception as e:
                 logger.error(f"Ошибка при получении потока ответа в консоли: {e}")
                 print("Ошибка: Не удалось получить ответ.", end="")
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
