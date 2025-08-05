import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
import time
import threading
import os
import logging
import pickle
from collections import defaultdict, deque
from gensim.models import KeyedVectors
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import requests
import gzip
import shutil

# === НАСТРОЙКИ ===
# Указываем полный путь, как у тебя
EMBEDDING_PATH = r"C:\Users\alex\Downloads\cc.ru.300.vec"
EMBEDDING_GZ_PATH = EMBEDDING_PATH + ".gz"
EMBEDDING_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
# Пример MD5 (реальный нужно вычислить после загрузки)
# Если не хочешь проверять — оставь как None
EXPECTED_MD5 = None  # или вставь реальный MD5

MAX_NODES = 5000
SLEEP_CYCLE = 15
FORGET_THRESHOLD = 0.1
ATTENTION_DECAY = 0.93
GENERATION_TEMP = 0.7
DISSONANCE_THRESHOLD = 0.4
SAVE_INTERVAL = 300  # автосохранение каждые 5 минут
MEMORY_HISTORY_LIMIT = 1000
MAX_CONTEXT_LENGTH = 10
PERSIST_FILE = "sin_state.pkl"

# === ЛОГГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sin.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIN")


# === ФУНКЦИИ ДЛЯ СКАЧИВАНИЯ И РАСПАКОВКИ ===
def calculate_md5(filepath):
    """Вычисление MD5 хэша файла"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Ошибка при вычислении MD5: {e}")
        return None


def download_embeddings(url, gz_path):
    """Скачивание файла по частям"""
    logger.info(f"Начинаю загрузку с {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(gz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (total_size // 10) == 0:
                            logger.info(f"Загрузка: {percent:.1f}%")

        logger.info(f"Файл сохранён: {gz_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке: {str(e)}")
        return False


def extract_gz(gz_path, vec_path):
    """Распаковка .gz в .vec"""
    logger.info("Распаковка архива...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Файл распакован: {vec_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при распаковке: {str(e)}")
        return False


def check_and_download_embeddings():
    """Проверяет наличие файла и скачивает при необходимости"""
    if os.path.exists(EMBEDDING_PATH):
        logger.info(f"Файл найден: {EMBEDDING_PATH}")
        if EXPECTED_MD5:
            file_md5 = calculate_md5(EMBEDDING_PATH)
            if file_md5 and file_md5.lower() == EXPECTED_MD5.lower():
                logger.info("✅ MD5 проверка пройдена.")
            else:
                logger.warning("MD5 не совпадает! Файл может быть повреждён.")
        return True

    logger.warning(f"Файл не найден: {EMBEDDING_PATH}")
    print("Файл эмбеддингов отсутствует. Начать загрузку? (y/n): ", end="")
    if input().lower() != 'y':
        logger.critical("Загрузка отменена.")
        return False

    # Скачиваем .gz
    if not download_embeddings(EMBEDDING_URL, EMBEDDING_GZ_PATH):
        logger.critical("Не удалось загрузить файл.")
        return False

    # Распаковываем
    if not extract_gz(EMBEDDING_GZ_PATH, EMBEDDING_PATH):
        logger.critical("Не удалось распаковать файл.")
        return False

    # Проверяем MD5, если задан
    if EXPECTED_MD5:
        file_md5 = calculate_md5(EMBEDDING_PATH)
        if file_md5 and file_md5.lower() == EXPECTED_MD5.lower():
            logger.info("✅ MD5 проверка пройдена.")
        else:
            logger.critical("MD5 не совпадает после загрузки!")
            return False

    logger.info("✅ Эмбеддинги готовы!")
    return True


# === СТРУКТУРЫ ДАННЫХ ===
@dataclass
class MemoryItem:
    vector: np.ndarray
    text: str
    level: int
    timestamp: float
    access_count: int = 0


@dataclass
class DialogContext:
    last_messages: deque = field(default_factory=lambda: deque(maxlen=MAX_CONTEXT_LENGTH))
    current_theme: Optional[str] = None
    thematic_attention: float = 0.0


# === ЯДРО СИСТЕМЫ ===
class RuEmbedder:
    def __init__(self, filepath=EMBEDDING_PATH):
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


class Sin:
    VERSION = "6.0"

    def __init__(self, persist_file: str = PERSIST_FILE):
        # Сначала проверяем и скачиваем эмбеддинги
        if not check_and_download_embeddings():
            raise RuntimeError("Не удалось подготовить эмбеддинги. Завершение.")
        self.embedder = RuEmbedder()
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
        self.persist_file = persist_file
        self.last_save_time = time.time()
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
                    'level_nodes': self.level_nodes
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

    def learn(self, text: str, from_dialog: bool = False) -> Dict:
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
                self.memory.append(MemoryItem(
                    vector=vec.copy(),
                    text=word,
                    level=0,
                    timestamp=time.time()
                ))
            if resonance < DISSONANCE_THRESHOLD and not from_dialog:
                question = self.generate_question(word)
                self.pending_questions.append(question)

        self.update_attention_weights(active_ids)
        total_vec /= len(words)
        self.memory.append(MemoryItem(
            vector=total_vec.copy(),
            text=' '.join(words),
            level=1,
            timestamp=time.time()
        ))

        self.form_hierarchy()

        activation = np.array([n.last_activation for n in self.nodes.values()])
        if len(activation) > 0:
            self.activation_history.append(activation.copy())
            if len(self.activation_history) > 50:
                self.activation_history.pop(0)

        self.cognitive_load = len(self.pending_questions) / 10 + len(self.memory) / 1000
        self.modulate_sleep()

        self.t += 1
        self._auto_save()

        return {"status": "learned", "response": f"Sin понял: '{text}'"}

    def generate_question(self, word: str) -> str:
        try:
            similar = self.embedder.model.most_similar(positive=[word], topn=1)
            return f"Я не до конца понимаю '{word}'. Это похоже на '{similar[0][0]}'? Или чем отличается?"
        except:
            return f"Что такое '{word}'? Можешь объяснить проще?"

    def form_hierarchy(self):
        level_0_nodes = [nid for nid in self.level_nodes[0] if self.nodes[nid].amplitude > 0.3]
        if len(level_0_nodes) > 2:
            combined_vec = np.mean([self.nodes[nid].pattern for nid in level_0_nodes], axis=0)
            combined_vec /= (np.linalg.norm(combined_vec) + 1e-8)
            active_ids = self.activate_input(combined_vec, level=1)
            for nid in level_0_nodes:
                for new_id in active_ids:
                    self.nodes[nid].connections[new_id] = 0.5
                    self.nodes[new_id].connections[nid] = 0.5

    def activate_input(self, vec, level=0, text=""):
        active_ids = []
        freq = 1.0 + np.linalg.norm(vec) * 0.5
        node = Resonator(self.node_counter, freq=freq, level=level)
        node.pattern = vec.copy()
        self.nodes[self.node_counter] = node
        node.excite(1.0)
        active_ids.append(self.node_counter)
        self.level_nodes[level].append(self.node_counter)
        self.node_counter += 1
        return active_ids

    def propagate_wave(self, source_id, amplitude, depth=0, max_depth=4):
        if depth >= max_depth or source_id not in self.nodes:
            return
        source = self.nodes[source_id]
        for target_id, strength in source.connections.items():
            if target_id in self.nodes:
                target = self.nodes[target_id]
                received_amp = amplitude * strength * 0.7
                if received_amp > 0.05:
                    target.excite(received_amp)
                    self.propagate_wave(target_id, received_amp, depth + 1, max_depth)

    def update_attention_weights(self, active_ids):
        for src_id in active_ids:
            src_node = self.nodes[src_id]
            for tgt_id in src_node.connections:
                if tgt_id in self.nodes:
                    tgt_node = self.nodes[tgt_id]
                    if self.are_in_phase(src_node, tgt_node):
                        delta = 0.2
                    else:
                        delta = 0.05
                    src_node.connections[tgt_id] = min(1.0, src_node.connections[tgt_id] + delta)
                    tgt_node.connections[src_id] = min(1.0, tgt_node.connections[src_id] + delta)
                    src_node.attention = min(1.0, src_node.attention + 0.05)
                    tgt_node.attention = min(1.0, tgt_node.attention + 0.05)

    def are_in_phase(self, node1, node2, tol=0.5):
        return abs((node1.phase - node2.phase) % (2 * np.pi)) < tol

    def modulate_sleep(self):
        if self.cognitive_load > 0.8 and not self.sleeping:
            self.start_sleep()

    def start_sleep(self):
        self.sleeping = True
        print("\n🌙 Sin засыпает... (когнитивная нагрузка: %.2f)" % self.cognitive_load)
        threading.Thread(target=self.dream_cycle, daemon=True).start()

    def dream_cycle(self):
        time.sleep(1)
        print("\n🧠 Sin видит сны...")
        for _ in range(5):
            if len(self.memory) == 0:
                continue
            mem = random.choice(self.memory)
            vec = mem.vector
            noise = np.random.normal(0, 0.05, vec.shape)
            dream = vec + noise
            dream /= (np.linalg.norm(dream) + 1e-8)
            dream_text = f"[сон:{mem.text}]"
            if self.check_resonance(dream) < 0.8:
                self.memory.append(MemoryItem(
                    vector=dream.copy(),
                    text=dream_text,
                    level=mem.level,
                    timestamp=self.t
                ))
            time.sleep(0.5)
        self.sleeping = False
        self.cognitive_load *= 0.5
        print("\n✨ Sin проснулся. Память укреплена.\n")

    def check_resonance(self, vec):
        sims = []
        for mem in self.memory:
            if mem.vector.shape != vec.shape:
                continue
            sim = cosine_similarity([vec], [mem.vector])[0][0]
            if sim > 0.2:
                sims.append(sim)
        return max(sims) if sims else 0.0

    def respond(self, text: str) -> str:
        if self.sleeping:
            if self.pending_questions:
                q = self.pending_questions.pop(0)
                return f"Во сне: '{q}'"
            return "Zzz... Sin спит."

        if self.pending_questions and random.random() < 0.3:
            return f"❓ {self.pending_questions.pop(0)}"

        words = self.tokenize(text)
        if not words:
            return "Я слушаю..."

        query_vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        best_sim = 0.0
        best_match = None
        for mem in self.memory:
            if mem.vector.shape != query_vec.shape:
                continue
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

    def status(self):
        return f"""
        🌐 Sin — Сеть Интуитивного Понимания v{self.VERSION}
        Время: {self.t}
        Узлов: {len(self.nodes)}
        Память: {len(self.memory)}
        Состояние: {'Спит' if self.sleeping else 'Бодрствует'}
        Уровни: {len(self.level_nodes[0])} слов, {len(self.level_nodes[1])} фраз
        """


# === ИНТЕРФЕЙСЫ ===
app = FastAPI(title=f"SIN API v{Sin.VERSION}")
sin = Sin()


@app.post("/learn")
async def api_learn(text: dict):
    result = sin.learn(text.get("text", ""))
    return JSONResponse(result)


@app.post("/respond")
async def api_respond(text: dict):
    response = sin.respond(text.get("text", ""))
    return {"response": response}


@app.get("/status")
async def api_status():
    return {
        "time": sin.t,
        "nodes": len(sin.nodes),
        "memory": len(sin.memory),
        "sleeping": sin.sleeping,
        "cognitive_load": sin.cognitive_load,
        "questions": len(sin.pending_questions)
    }


def run_telegram(token: str):
    bot = Application.builder().token(token).build()

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f"Привет! Я SIN v{Sin.VERSION}. Давай пообщаемся!")

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        response = sin.respond(user_text)
        await update.message.reply_text(f"💬 Sin: {response}")

    bot.add_handler(CommandHandler("start", start))
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    bot.run_polling()


# === ЗАПУСК ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "telegram", "cli"], default="cli")
    parser.add_argument("--token", type=str, help="Telegram токен")
    args = parser.parse_args()

    if args.mode == "api":
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.mode == "telegram":
        token = args.token or os.getenv("TELEGRAM_TOKEN")
        if not token:
            print("Требуется токен Telegram")
        else:
            run_telegram(token)
    else:
        print(sin.status())
        print("\nДоступные команды:")
        print("  введи текст — Sin ответит")
        print("  !status — статус")
        print("  !sleep — заставить поспать")
        print("  !visualize — график резонанса")
        print("  !quit — выход\n")

        while True:
            try:
                user_input = input("> Sin, ").strip()
                if user_input.lower() == "!quit":
                    break
                elif user_input.lower() == "!status":
                    print(sin.status())
                elif user_input.lower() == "!sleep":
                    sin.start_sleep()
                elif user_input.lower() == "!visualize":
                    if sin.activation_history:
                        data = np.array(sin.activation_history)
                        plt.imshow(data.T, aspect='auto', cmap='plasma', interpolation='none')
                        plt.colorbar()
                        plt.title("Волны резонанса")
                        plt.show()
                    else:
                        print("Нет данных для визуализации.")
                else:
                    learn_result = sin.learn(user_input)
                    print(learn_result["response"])
                    response = sin.respond(user_input)
                    print(f"💬 Sin: {response}")
            except KeyboardInterrupt:
                break
