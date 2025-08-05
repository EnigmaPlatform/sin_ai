import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
import time
import threading
from queue import Queue
import os
from gensim.models import KeyedVectors
from collections import defaultdict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# === НАСТРОЙКИ ===
EMBEDDING_FILE = "cc.ru.300.vec"
MAX_NODES = 1000
SLEEP_CYCLE = 10
FORGET_THRESHOLD = 0.1
ATTENTION_DECAY = 0.95
GENERATION_TEMP = 0.7
DISSONANCE_THRESHOLD = 0.4  # Ниже — возникает вопрос
TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"

# === УЛУЧШЕННЫЙ RuEmbedder ===
class RuEmbedder:
    def __init__(self, filepath=EMBEDDING_FILE):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}\n"
                                  f"Скачай с: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz\n"
                                  f"Распакуй и положи в эту папку как 'cc.ru.300.vec'")
        print("🌀 Загрузка русских эмбеддингов (FastText, 300d)...")
        self.model = KeyedVectors.load_word2vec_format(filepath, binary=False, limit=200000)
        self.dim = self.model.vector_size
        print(f"✅ Загружено {len(self.model.key_to_index)} слов")

    def get_vector(self, word):
        word_clean = word.lower().strip(".,!?\"'()[]{}:;—-")
        if not word_clean:
            return np.zeros(self.dim)
        if word_clean in self.model:
            return self.model[word_clean].copy()
        try:
            similar = self.model.most_similar(positive=[word_clean], topn=1)
            print(f"⚠️ '{word}' не найдено. Используем: '{similar[0][0]}'")
            return self.model[similar[0][0]].copy()
        except:
            print(f"⚠️ '{word}' неизвестно. Используем нейтральный вектор.")
            return np.random.normal(0, 0.1, self.dim)

    def similarity(self, word1, word2):
        w1 = word1.lower().strip(".,!?\"'()")
        w2 = word2.lower().strip(".,!?\"'()")
        if w1 in self.model and w2 in self.model:
            return self.model.similarity(w1, w2)
        return 0.0

    def generate_next_word(self, word, top_k=5, temp=GENERATION_TEMP):
        if word not in self.model:
            return None
        similar = self.model.most_similar(positive=[word], topn=top_k * 2)
        words, scores = zip(*similar)
        scores = np.array(scores) ** (1 / temp)
        probs = scores / scores.sum()
        return np.random.choice(words, p=probs)


# === УЛУЧШЕННЫЙ РЕЗОНАТОР С ФАЗОЙ ===
class Resonator:
    def __init__(self, node_id, freq=1.0, phase=0.0, damping=0.1, level=0):
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
        self.phase_history = []  # Для анализа синхронизации

    def excite(self, amp, phase_offset=0.0):
        self.amplitude = amp * self.attention
        self.phase = phase_offset
        self.last_activation = amp

    def step(self, dt=0.1):
        if self.amplitude > 0.01:
            self.phase += self.freq * dt
            self.phase = self.phase % (2 * np.pi)
            self.amplitude *= (1 - self.damping * dt)
            self.attention *= ATTENTION_DECAY
        else:
            self.amplitude = 0.0


# === SIN v3.0 — ПОЛНЫЙ КОГНИТИВНЫЙ АГЕНТ ===
class Sin:
    def __init__(self):
        self.embedder = RuEmbedder(EMBEDDING_FILE)
        self.nodes = {}
        self.node_counter = 0
        self.memory_interference = []
        self.activation_history = []
        self.t = 0
        self.sleeping = False
        self.hierarchy_levels = 3
        self.level_nodes = [[] for _ in range(self.hierarchy_levels)]
        self.recent_context = []
        self.word_frequency = defaultdict(int)
        self.cognitive_load = 0.0  # Для модуляции сна
        self.pending_questions = []

    def tokenize(self, text):
        words = [word.strip(".,!?\"'()[]{}:;—-") for word in text.lower().split() if word.isalpha()]
        for word in words:
            self.word_frequency[word] += 1
        return words

    def are_in_phase(self, node1, node2, tol=0.5):
        """Проверка фазовой синхронизации"""
        return abs((node1.phase - node2.phase) % (2 * np.pi)) < tol

    def form_concept_from_phase_sync(self):
        """Объединение синхронных узлов в концепт (уровень 1)"""
        active_nodes = [n for n in self.nodes.values() if n.amplitude > 0.3]
        synced_groups = []
        for node in active_nodes:
            matched = False
            for group in synced_groups:
                if self.are_in_phase(node, group[0]):
                    group.append(node)
                    matched = True
                    break
            if not matched:
                synced_groups.append([node])

        for group in synced_groups:
            if len(group) > 2:
                combined_vec = np.mean([n.pattern for n in group], axis=0)
                combined_vec /= (np.linalg.norm(combined_vec) + 1e-8)
                new_id = self.node_counter
                new_node = Resonator(new_id, level=1)
                new_node.pattern = combined_vec.copy()
                self.nodes[new_id] = new_node
                self.level_nodes[1].append(new_id)
                self.node_counter += 1
                for member in group:
                    self.nodes[member.id].connections[new_id] = 0.6
                    self.nodes[new_id].connections[member.id] = 0.6

    def hierarchical_forget(self):
        """Иерархическое забывание: низкие уровни забываются чаще"""
        if len(self.memory_interference) < 500:
            return
        to_remove = []
        for i, mem in enumerate(self.memory_interference):
            if isinstance(mem['text'], str):
                words = self.tokenize(mem['text'])
                freq_score = sum(self.word_frequency.get(w, 0) for w in words) / (len(words) + 1e-8)
                # Чем выше уровень — тем меньше шанса быть забытым
                forget_bias = 0.5 if mem['level'] == 0 else 0.1
                if freq_score < FORGET_THRESHOLD * forget_bias:
                    to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.memory_interference.pop(i)
        if to_remove:
            print(f"🧹 Забыто {len(to_remove)} элементов (иерархически)")

    def generate_question(self, text):
        """Генерация внутреннего вопроса при диссонансе"""
        words = self.tokenize(text)
        if not words:
            return "Что это значит?"
        seed = random.choice(words)
        try:
            similar = self.embedder.model.most_similar(positive=[seed], topn=1)
            return f"Похоже на '{similar[0][0]}'... Но чем отличается?"
        except:
            return f"Что такое '{seed}'? Как это связано с другими?"

    def modulate_sleep(self):
        """Сон при высокой когнитивной нагрузке"""
        if self.cognitive_load > 0.8 and not self.sleeping:
            self.start_sleep()

    def update_attention_weights(self, active_ids):
        for src_id in active_ids:
            src_node = self.nodes[src_id]
            for tgt_id in src_node.connections:
                if tgt_id in self.nodes:
                    tgt_node = self.nodes[tgt_id]
                    if self.are_in_phase(src_node, tgt_node):
                        delta = 0.2  # Сильнее усиливаем при синхронизации
                    else:
                        delta = 0.05
                    src_node.connections[tgt_id] = min(1.0, src_node.connections[tgt_id] + delta)
                    tgt_node.connections[src_id] = min(1.0, tgt_node.connections[src_id] + delta)
                    src_node.attention = min(1.0, src_node.attention + 0.05)
                    tgt_node.attention = min(1.0, tgt_node.attention + 0.05)

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
                received_amp = amplitude * strength * (0.5 + 0.5 * target.attention)
                if received_amp > 0.05:
                    target.excite(received_amp)
                    self.propagate_wave(target_id, received_amp, depth + 1, max_depth)

    def check_resonance(self, vec):
        sims = []
        for mem in self.memory_interference:
            if mem['vector'].shape != vec.shape:
                continue
            sim = cosine_similarity([vec], [mem['vector']])[0][0]
            if sim > 0.2:
                sims.append(sim)
        return max(sims) if sims else 0.0

    def learn(self, text):
        if self.sleeping:
            return {"status": "sleeping", "response": "Zzz... Sin спит."}

        words = self.tokenize(text)
        if not words:
            return {"status": "empty", "response": "Пустой ввод."}

        self.recent_context.append(text)
        self.hierarchical_forget()

        total_vec = np.zeros(self.embedder.dim)
        active_ids = []
        for word in words:
            vec = self.embedder.get_vector(word)
            total_vec += vec
            resonance = self.check_resonance(vec)
            new_ids = self.activate_input(vec, level=0, text=word)
            active_ids.extend(new_ids)
            for nid in new_ids:
                self.propagate_wave(nid, 1.0)
            if resonance < 0.6:
                self.memory_interference.append({
                    'vector': vec.copy(),
                    'text': word,
                    'level': 0,
                    'timestamp': self.t
                })
            if resonance < DISSONANCE_THRESHOLD:
                question = self.generate_question(word)
                self.pending_questions.append(question)

        self.update_attention_weights(active_ids)
        total_vec /= len(words)
        self.memory_interference.append({
            'vector': total_vec.copy(),
            'text': ' '.join(words),
            'level': 1,
            'timestamp': self.t
        })

        self.form_concept_from_phase_sync()

        activation = np.array([n.last_activation for n in self.nodes.values()])
        if len(activation) > 0:
            self.activation_history.append(activation.copy())
            if len(self.activation_history) > 50:
                self.activation_history.pop(0)

        self.cognitive_load = len(self.pending_questions) / 10 + len(self.memory_interference) / 1000
        self.modulate_sleep()

        self.t += 1
        return {"status": "learned", "response": f"Sin понял: '{text}'"}

    def start_sleep(self):
        self.sleeping = True
        print("\n🌙 Sin засыпает... (когнитивная нагрузка: %.2f)" % self.cognitive_load)
        threading.Thread(target=self.dream_cycle, daemon=True).start()

    def dream_cycle(self):
        time.sleep(1)
        print("\n🧠 Sin видит сны...")
        for _ in range(5):
            if len(self.memory_interference) == 0:
                continue
            mem = random.choice(self.memory_interference)
            vec = mem['vector']
            noise = np.random.normal(0, 0.05, vec.shape)
            dream = vec + noise
            dream /= (np.linalg.norm(dream) + 1e-8)
            dream_text = f"[сон:{mem['text']}]"
            if self.check_resonance(dream) < 0.8:
                self.memory_interference.append({
                    'vector': dream.copy(),
                    'text': dream_text,
                    'level': mem['level'],
                    'timestamp': self.t
                })
            time.sleep(0.5)
        self.sleeping = False
        self.cognitive_load *= 0.5
        print("\n✨ Sin проснулся. Память укреплена.\n")

    def respond(self, text):
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
        for mem in self.memory_interference:
            if mem['vector'].shape != query_vec.shape:
                continue
            sim = cosine_similarity([query_vec], [mem['vector']])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_match = mem['text']

        if best_sim > 0.6:
            hints = ["Это напоминает мне о", "Я чувствую сходство с"]
            return f"{random.choice(hints)} '{best_match}' (схожесть: {best_sim:.2f})."
        elif best_sim > 0.4:
            return f"Частично понимаю. Ещё не до конца ясно ({best_sim:.2f})."
        else:
            return f"Новое. Ещё не резонирует. Расскажи больше."


# === SIN API (FastAPI) ===
sin = Sin()
app = FastAPI(title="Sin API", description="Сеть Интуитивного Понимания")

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
        "memory": len(sin.memory_interference),
        "sleeping": sin.sleeping,
        "cognitive_load": sin.cognitive_load,
        "questions": len(sin.pending_questions)
    }


# === TELEGRAM-БОТ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я Sin — сеть, которая учится на резонансе. Напиши мне что-нибудь!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    learn_result = sin.learn(user_text)
    response = sin.respond(user_text)
    await update.message.reply_text(f"💬 Sin: {response}")

def run_telegram():
    app_bot = Application.builder().token(TELEGRAM_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_bot.run_polling()

# === ЗАПУСК ===
if __name__ == "__main__":
    import multiprocessing
    p1 = multiprocessing.Process(target=uvicorn.run, args=(app,), kwargs={"host": "127.0.0.1", "port": 8000})
    p2 = multiprocessing.Process(target=run_telegram)
    p1.start()
    p2.start()
    print("🚀 Sin запущен: API на http://127.0.0.1:8000, Telegram-бот активен.")
    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("\nSin отключается...")
