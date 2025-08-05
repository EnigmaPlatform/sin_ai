import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import random
import time
import threading
import os
import logging
import pickle
from collections import defaultdict, deque
from gensim.models import KeyedVectors
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import uvicorn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import asyncio
import aiofiles
import aiohttp
import psutil
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import streamlit as st
from vispy import app as vispy_app, scene, gloo
from vispy.scene import visuals as scene_visuals
from vispy.color import Color

# === НАСТРОЙКИ ===
EMBEDDING_FILE = "cc.ru.300.vec"
MAX_NODES = 5000
SLEEP_CYCLE = 15
FORGET_THRESHOLD = 0.1
ATTENTION_DECAY = 0.93
GENERATION_TEMP = 0.7
DISSONANCE_THRESHOLD = 0.4
SAVE_INTERVAL = 300
MEMORY_HISTORY_LIMIT = 1000
MAX_CONTEXT_LENGTH = 10
PERSIST_FILE = "sin_state.pkl"
LOG_FILE = "sin.log"
TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"
VISPY_SIZE = (1600, 900)

# === ЛОГГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIN")


# === Pydantic МОДЕЛИ (V2) ===
class LearnRequest(BaseModel):
    text: str = Field(..., max_length=500)

    @field_validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class RespondRequest(BaseModel):
    text: str = Field(..., max_length=500)


class GenerateRequest(BaseModel):
    seed: str
    length: int = Field(5, ge=1, le=20)


# === СТРУКТУРЫ ДАННЫХ ===
@dataclass
class MemoryItem:
    vector: np.ndarray
    text: str
    level: int
    timestamp: float
    access_count: int = 0
    phase_cluster_id: Optional[int] = None
    reward_score: float = 0.0


@dataclass
class DialogContext:
    last_messages: deque = field(default_factory=lambda: deque(maxlen=MAX_CONTEXT_LENGTH))
    current_theme: Optional[str] = None
    thematic_attention: float = 0.0


# === RuEmbedder (обновлённый) ===
class RuEmbedder:
    def __init__(self, filepath=EMBEDDING_FILE):
        self.model = self._load_embeddings(filepath)
        self.dim = self.model.vector_size
        logger.info(f"Загружено {len(self.model.key_to_index)} слов (dim={self.dim})")

    def _load_embeddings(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл эмбеддингов не найден: {filepath}")
        logger.info("🌀 Загрузка эмбеддингов...")
        return KeyedVectors.load_word2vec_format(filepath, binary=False, limit=300000)

    def get_vector(self, word: str) -> np.ndarray:
        word_clean = word.lower().strip(".,!?\"'()[]{}:;—-")
        if not word_clean:
            return np.zeros(self.dim)
        if word_clean in self.model:
            return self.model[word_clean].copy()
        try:
            similar = self.model.most_similar(positive=[word_clean], topn=1)
            logger.debug(f"Слово '{word}' заменено на '{similar[0][0]}'")
            return self.model[similar[0][0]].copy()
        except:
            logger.debug(f"Неизвестное слово: '{word}'")
            return np.random.normal(0, 0.1, self.dim)

    def generate_sequence(self, seed_word: str, length=5, diversity=1.0) -> List[str]:
        sequence = [seed_word]
        current_word = seed_word
        for _ in range(length - 1):
            try:
                similar = self.model.most_similar(positive=[current_word], topn=10)
                words, scores = zip(*similar)
                probs = np.array(scores) ** (1 / diversity)
                probs /= probs.sum()
                next_word = np.random.choice(words, p=probs)
                sequence.append(next_word)
                current_word = next_word
            except:
                break
        return sequence


# === Resonator с фазой и вниманием ===
class Resonator:
    __slots__ = ['id', 'freq', 'phase', 'amplitude', 'damping', 'connections',
                 'pattern', 'level', 'last_activation', 'attention', 'phase_history']
    def __init__(self, node_id: int, freq: float = 1.0, phase: float = 0.0,
                 damping: float = 0.1, level: int = 0):
        self.id = node_id
        self.freq = freq
        self.phase = phase
        self.amplitude = 0.0
        self.damping = damping
        self.connections = defaultdict(float)
        self.pattern = None
        self.level = level
        self.last_activation = 0.0
        self.attention = 1.0
        self.phase_history = deque(maxlen=100)

    def excite(self, amp: float, phase_offset: float = 0.0):
        self.amplitude = amp * self.attention
        self.phase = phase_offset
        self.last_activation = amp
        self.phase_history.append(self.phase)

    def step(self, dt: float = 0.1):
        if self.amplitude > 0.01:
            self.phase += self.freq * dt
            self.phase %= (2 * np.pi)
            self.amplitude *= (1 - self.damping * dt)
            self.attention *= ATTENTION_DECAY
            self.phase_history.append(self.phase)
        else:
            self.amplitude = 0.0

    def update_connection(self, target_id: int, delta: float):
        self.connections[target_id] = np.clip(self.connections[target_id] + delta, 0.1, 1.0)


# === SIN v10.0 — Финальная версия ===
class Sin:
    VERSION = "10.0"
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_file=PERSIST_FILE):
        if hasattr(self, 'initialized'):
            return
        self.persist_file = persist_file
        self.embedder = RuEmbedder(EMBEDDING_FILE)
        self.nodes = {}
        self.node_counter = 0
        self.memory = []
        self.activation_history = deque(maxlen=100)
        self.t = 0
        self.sleeping = False
        self.hierarchy_levels = 3
        self.level_nodes = [[] for _ in range(self.hierarchy_levels)]
        self.word_frequency = defaultdict(int)
        self.cognitive_load = 0.0
        self.pending_questions = []
        self.dialog_context = DialogContext()
        self.last_save_time = time.time()
        self.phase_clusters = []
        self.request_count = 0
        self.error_count = 0
        self.save_lock = asyncio.Lock()
        self.learn_lock = asyncio.Lock()
        self.cluster_labels = []
        self.rl_policy = {"ask_question": 0.7, "generate": 0.5}
        self._init_system()
        logger.info(f"Система SIN v{self.VERSION} инициализирована")

    def _init_system(self):
        if os.path.exists(self.persist_file):
            self._load_state()
        else:
            logger.info("Создана новая модель")

    def _load_state(self):
        try:
            with open(self.persist_file, 'rb') as f:
                data = pickle.load(f)
                self.__dict__.update(data)
            logger.info(f"Состояние загружено из {self.persist_file}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")

    def save_state(self):
        try:
            with open(self.persist_file, 'wb') as f:
                pickle.dump({
                    'nodes': self.nodes,
                    'node_counter': self.node_counter,
                    'memory': self.memory,
                    't': self.t,
                    'word_frequency': self.word_frequency,
                    'level_nodes': self.level_nodes,
                    'phase_clusters': self.phase_clusters,
                    'cluster_labels': self.cluster_labels,
                    'rl_policy': self.rl_policy
                }, f)
            logger.info(f"Состояние сохранено в {self.persist_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")

    def _auto_save(self):
        if time.time() - self.last_save_time > SAVE_INTERVAL:
            self.save_state()
            self.last_save_time = time.time()

    def tokenize(self, text: str) -> List[str]:
        words = [word.strip(".,!?\"'()[]{}:;—-") for word in text.lower().split() if word.isalpha()]
        for word in words:
            self.word_frequency[word] += 1
        return words

    def _update_dialog_context(self, text: str):
        self.dialog_context.last_messages.append(text)
        if len(self.dialog_context.last_messages) >= 3:
            recent_text = " ".join(self.dialog_context.last_messages)
            theme_vector = np.mean([self.embedder.get_vector(w) for w in self.tokenize(recent_text)], axis=0)
            if self.dialog_context.current_theme is None:
                self.dialog_context.current_theme = hashlib.md5(theme_vector.tobytes()).hexdigest()
                self.dialog_context.thematic_attention = 0.5
            else:
                old_theme_vec = self.embedder.get_vector(self.dialog_context.current_theme[:10])
                similarity = cosine_similarity([theme_vector], [old_theme_vec])[0][0]
                self.dialog_context.thematic_attention = 0.3 * self.dialog_context.thematic_attention + 0.7 * similarity

    def are_in_phase(self, node1: Resonator, node2: Resonator, tol=0.5) -> bool:
        return abs((node1.phase - node2.phase) % (2 * np.pi)) < tol

    def assign_to_phase_cluster(self, node: Resonator) -> int:
        for cluster_id, cluster in enumerate(self.phase_clusters):
            if cluster and self.are_in_phase(node, self.nodes[cluster[0]]):
                cluster.append(node.id)
                return cluster_id
        new_cluster = [node.id]
        self.phase_clusters.append(new_cluster)
        return len(self.phase_clusters) - 1

    def hierarchical_forget(self):
        if len(self.memory) < 1000:
            return
        to_remove = []
        for i, mem in enumerate(self.memory):
            if isinstance(mem.text, str):
                words = self.tokenize(mem.text)
                freq_score = sum(self.word_frequency.get(w, 0) for w in words) / (len(words) + 1e-8)
                forget_bias = 0.5 if mem.level == 0 else 0.1
                if freq_score < FORGET_THRESHOLD * forget_bias:
                    to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.memory.pop(i)
        if to_remove:
            logger.info(f"🧹 Иерархически забыто {len(to_remove)} элементов")

    def generate_question(self, word: str) -> str:
        try:
            similar = self.embedder.model.most_similar(positive=[word], topn=1)
            return f"Я не до конца понимаю '{word}'. Это похоже на '{similar[0][0]}'? Или чем отличается?"
        except:
            return f"Что такое '{word}'? Можешь объяснить проще?"

    def learn(self, text: str, from_dialog: bool = False, user_feedback: str = "neutral") -> Dict:
        self._update_dialog_context(text)
        words = self.tokenize(text)
        if not words:
            return {"status": "empty", "response": "Пустой ввод"}

        total_vec = np.zeros(self.embedder.dim)
        active_ids = []
        for word in words:
            vec = self.embedder.get_vector(word)
            total_vec += vec
            resonance = self.check_resonance(vec)
            new_id = self.node_counter
            node = Resonator(new_id, level=0)
            node.pattern = vec.copy()
            self.nodes[new_id] = node
            node.excite(1.0)
            active_ids.append(new_id)
            self.node_counter += 1

            if resonance < 0.6:
                cluster_id = self.assign_to_phase_cluster(node)
                self.memory.append(MemoryItem(
                    vector=vec.copy(),
                    text=word,
                    level=0,
                    timestamp=time.time(),
                    phase_cluster_id=cluster_id
                ))
            if resonance < DISSONANCE_THRESHOLD and not from_dialog:
                question = self.generate_question(word)
                self.pending_questions.append(question)

        total_vec /= len(words)
        self.memory.append(MemoryItem(
            vector=total_vec.copy(),
            text=' '.join(words),
            level=1,
            timestamp=time.time()
        ))

        self.hierarchical_forget()
        self._auto_save()

        # === RL: обучение с подкреплением ===
        if user_feedback == "good":
            self.rl_policy["ask_question"] += 0.1
            self.rl_policy["generate"] += 0.1
        elif user_feedback == "bad":
            self.rl_policy["ask_question"] -= 0.1
            self.rl_policy["generate"] -= 0.1

        return {"status": "learned", "response": f"Sin понял: '{text}'"}

    def respond(self, text: str) -> str:
        if self.pending_questions and random.random() < self.rl_policy["ask_question"]:
            return f"❓ {self.pending_questions.pop(0)}"
        if self.sleeping:
            return "Zzz... Sin спит."
        words = self.tokenize(text)
        if not words:
            return "Я слушаю..."
        query_vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        best_sim = 0.0
        best_match = None
        for mem in self.memory:
            sim = cosine_similarity([query_vec], [mem.vector])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_match = mem.text
        if best_sim > 0.6:
            hints = ["Это напоминает мне о", "Я чувствую сходство с"]
            return f"{random.choice(hints)} '{best_match}' (схожесть: {best_sim:.2f})."
        elif best_sim > 0.4:
            return f"Частично понимаю. Ещё не до конца ясно ({best_sim:.2f})."
        else:
            return f"Новое. Ещё не резонирует. Расскажи больше."

    def check_resonance(self, vec):
        sims = []
        for mem in self.memory:
            sim = cosine_similarity([vec], [mem.vector])[0][0]
            if sim > 0.2:
                sims.append(sim)
        return max(sims) if sims else 0.0

    def generate_response(self, seed: str, length=5) -> str:
        base_sequence = self.embedder.generate_sequence(seed, length=length)
        enhanced = []
        for word in base_sequence:
            try:
                mem_sim = max(
                    (cosine_similarity([self.embedder.get_vector(word)], [m.vector])[0][0], m.text)
                    for m in self.memory
                )
                if mem_sim[0] > 0.6:
                    enhanced.append(mem_sim[1])
                else:
                    enhanced.append(word)
            except:
                enhanced.append(word)
        return " ".join(enhanced[:length])


# === 3D ВИЗУАЛИЗАЦИЯ С ВОЛНАМИ РЕЗОНАНСА ===
class Sin3DVisualizer:
    def __init__(self, sin_instance):
        self.sin = sin_instance
        self.canvas = scene.SceneCanvas(size=VISPY_SIZE, title='Sin — 3D Когнитивный Ландшафт', show=True, bgcolor=Color('#111111'))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 15

        self.nodes = scene_visuals.Markers(parent=self.view.scene)
        self.connections = scene_visuals.Line(parent=self.view.scene)
        self.clusters = []

        self.node_data = np.zeros(MAX_NODES, dtype=[('position', float, 3), ('color', float, 4), ('size', float, 1)])
        self.connection_data = np.zeros(5000, dtype=[('start', float, 3), ('end', float, 3), ('color', float, 4)])

        self._init_data()
        self.timer = vispy_app.Timer('auto', connect=self.update, start=True)

    def _init_data(self):
        np.random.seed(42)
        self.node_data['position'] = np.random.randn(MAX_NODES, 3) * 2.0
        self.node_data['color'] = [0.3, 0.3, 0.3, 0.6]
        self.node_data['size'] = 5

    def update(self, event):
        self._sync_with_sin()
        self._update_nodes()
        self._update_connections()
        self._update_clusters()

    def _sync_with_sin(self):
        memory_items = self.sin.memory[:MAX_NODES]
        for i, mem in enumerate(memory_items):
            vec_3d = mem.vector[:3]
            self.node_data['position'][i] = vec_3d * 2.0

            level = mem.level
            hue = 0.6 if level == 0 else 0.3 if level == 1 else 0.0
            self.node_data['color'][i] = [hue, 0.7, 0.8, 0.9]

            size = 5 + 15 * mem.access_count / 10
            self.node_data['size'][i] = size

    def _update_nodes(self):
        self.nodes.set_data(pos=self.node_data['position'], face_color=self.node_data['color'], size=self.node_data['size'])

    def _update_connections(self):
        active_nodes = np.where(self.node_data['size'] > 8)[0][:100]
        connections = []
        for i in active_nodes:
            for j in active_nodes:
                if i >= j: continue
                dist = np.linalg.norm(self.node_data['position'][i] - self.node_data['position'][j])
                if dist < 3.0:
                    phase_diff = abs((self.sin.nodes.get(i, Resonator(0)).phase - self.sin.nodes.get(j, Resonator(0)).phase) % (2*np.pi))
                    hue = 0.6 if phase_diff < 0.5 else 0.1
                    connections.append((self.node_data['position'][i], self.node_data['position'][j], [hue, 0.5, 1.0, 0.3]))
                if len(connections) >= 5000: break
            if len(connections) >= 5000: break

        if connections:
            data = np.array(connections, dtype=self.connection_data.dtype)
            self.connection_data[:len(data)] = data
            pos = np.vstack([(d[0], d[1]) for d in data])
            color = np.vstack([d[2] for d in data])
            self.connections.set_data(pos=pos, color=color, width=1.0)

    def _update_clusters(self):
        for cluster in self.clusters:
            cluster.parent = None
        self.clusters = []

        active_positions = np.array([d['position'] for d in self.node_data if d['size'] > 10])
        if len(active_positions) > 5:
            clustering = DBSCAN(eps=2.5, min_samples=3).fit(active_positions)
            labels = clustering.labels_
            for label in set(labels) - {-1}:
                cluster_points = active_positions[labels == label]
                center = np.mean(cluster_points, axis=0)
                radius = np.std(np.linalg.norm(cluster_points - center, axis=1)) + 0.5
                sphere = scene_visuals.Sphere(radius=radius, method='ico', parent=self.view.scene, color=[0.2, 0.6, 0.8, 0.1])
                sphere.transform = scene.transforms.STTransform(translate=center)
                self.clusters.append(sphere)

    def run(self):
        vispy_app.run()


# === ВЕБ-ИНТЕРФЕЙС (Streamlit) ===
def run_web(sin_instance):
    st.set_page_config(page_title="Sin — Когнитивный агент", layout="wide")
    st.title("🧠 Sin v10.0 — Сеть Интуитивного Понимания")

    if st.button("Перезагрузить Sin"):
        sin_instance._init_system()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💬 Диалог")
        user_input = st.text_input("Ты:")
        if st.button("Отправить"):
            if user_input:
                result = sin_instance.learn(user_input)
                response = sin_instance.respond(user_input)
                st.session_state.chat.append(("Ты", user_input))
                st.session_state.chat.append(("Sin", response))
        for speaker, text in st.session_state.get('chat', []):
            st.write(f"**{speaker}**: {text}")

        feedback = st.radio("Оцените ответ:", ["neutral", "good", "bad"])
        if st.button("Отправить оценку"):
            sin_instance.learn("feedback", user_feedback=feedback)

    with col2:
        st.subheader("📊 Визуализация")
        if sin_instance.activation_history:
            data = np.array(sin_instance.activation_history)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(data.T, aspect='auto', cmap='plasma', interpolation='none')
            ax.set_title("Волны резонанса")
            st.pyplot(fig)

        if len(sin_instance.cluster_labels) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(range(len(sin_instance.cluster_labels)), sin_instance.cluster_labels, c=sin_instance.cluster_labels, cmap='tab10')
            ax.set_title("Кластеризация памяти")
            st.pyplot(fig)


# === API ===
app = FastAPI(title=f"SIN API v{Sin.VERSION}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app.router.lifespan_context = lifespan

sin = Sin()

@app.post("/learn")
async def api_learn(request: LearnRequest):
    try:
        result = sin.learn(request.text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/respond")
async def api_respond(request: RespondRequest):
    try:
        response = sin.respond(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def api_generate(request: GenerateRequest):
    try:
        text = sin.generate_response(request.seed, request.length)
        return {"generated": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === TELEGRAM-БОТ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Привет! Я SIN v{Sin.VERSION}. Давай пообщаемся!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = sin.respond(user_text)
    await update.message.reply_text(f"💬 Sin: {response}")

def run_telegram():
    app_bot = Application.builder().token(TELEGRAM_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_bot.run_polling()


# === ЗАПУСК ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "telegram", "web", "3d"], default="web")
    args = parser.parse_args()

    if args.mode == "3d":
        visualizer = Sin3DVisualizer(sin)
        visualizer.run()
    elif args.mode == "web":
        st.session_state.chat = []
        run_web(sin)
    elif args.mode == "api":
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.mode == "telegram":
        run_telegram()
