import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
import time
import threading
from queue import Queue
import os
from gensim.models import KeyedVectors


# === НАСТРОЙКИ ===
EMBEDDING_FILE = "cc.ru.300.vec"  # Убедись, что файл здесь
MAX_NODES = 1000
SLEEP_CYCLE = 10


# === ЗАГРУЗКА РУССКИХ ЭМБЕДДИНГОВ (FastText Facebook) ===
class RuEmbedder:
    def __init__(self, filepath=EMBEDDING_FILE):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}\n"
                                  f"Скачай с: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz\n"
                                  f"Распакуй и положи в эту папку как 'cc.ru.300.vec'")

        print("🌀 Загрузка русских эмбеддингов (FastText, 300d)...")
        # Загружаем только ключи и векторы (не обучаем)
        self.model = KeyedVectors.load_word2vec_format(filepath, binary=False, limit=200000)
        self.dim = self.model.vector_size  # 300
        print(f"✅ Загружено {len(self.model.key_to_index)} слов")

    def get_vector(self, word):
        word_clean = word.lower().strip(".,!?\"'()[]{}:;—-")
        if not word_clean:
            return np.zeros(self.dim)

        if word_clean in self.model:
            return self.model[word_clean].copy()

        # Попробуем найти похожее слово
        try:
            similar = self.model.most_similar(positive=[word_clean], topn=1)
            print(f"⚠️ '{word}' не найдено. Используем: '{similar[0][0]}'")
            return self.model[similar[0][0]].copy()
        except:
            print(f"⚠️ '{word}' неизвестно. Используем нейтральный вектор.")
            return np.random.normal(0, 0.1, self.dim)  # слабый случайный вектор

    def similarity(self, word1, word2):
        w1 = word1.lower().strip(".,!?\"'()")
        w2 = word2.lower().strip(".,!?\"'()")
        if w1 in self.model and w2 in self.model:
            return self.model.similarity(w1, w2)
        return 0.0


# === РЕЗОНАТОР ===
class Resonator:
    def __init__(self, node_id, freq=1.0, phase=0.0, damping=0.1, level=0):
        self.id = node_id
        self.freq = freq
        self.phase = phase
        self.amplitude = 0.0
        self.damping = damping
        self.connections = {}
        self.pattern = None
        self.level = level
        self.last_activation = 0.0

    def excite(self, amp, phase_offset=0.0):
        self.amplitude = amp
        self.phase = phase_offset
        self.last_activation = amp

    def step(self, dt=0.1):
        if self.amplitude > 0.01:
            self.phase += self.freq * dt
            self.phase = self.phase % (2 * np.pi)
            self.amplitude *= (1 - self.damping * dt)
        else:
            self.amplitude = 0.0


# === SIN — СЕТЬ ИНТУИТИВНОГО ПОНИМАНИЯ (РУССКАЯ ВЕРСИЯ) ===
class Sin:
    def __init__(self, embedder):
        self.embedder = embedder
        self.nodes = {}
        self.node_counter = 0
        self.memory_interference = []
        self.activation_history = []
        self.t = 0
        self.sleeping = False
        self.hierarchy_levels = 3
        self.level_nodes = [[] for _ in range(self.hierarchy_levels)]
        self.recent_context = []

    def tokenize(self, text):
        return [word.strip(".,!?\"'()[]{}:;—-") for word in text.lower().split() if word.isalpha()]

    def activate_input(self, vec, level=0):
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

    def check_resonance(self, vec):
        sims = []
        for mem in self.memory_interference:
            if mem['vector'].shape != vec.shape:
                continue
            sim = cosine_similarity([vec], [mem['vector']])[0][0]
            if sim > 0.2:
                sims.append(sim)
        return max(sims) if sims else 0.0

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

    def learn(self, text):
        if self.sleeping:
            return "Zzz... Sin спит. Приходите позже."

        words = self.tokenize(text)
        if not words:
            return "Пустой ввод."

        self.recent_context.append(text)

        total_vec = np.zeros(self.embedder.dim)
        for word in words:
            vec = self.embedder.get_vector(word)
            total_vec += vec
            resonance = self.check_resonance(vec)
            active_ids = self.activate_input(vec, level=0)

            for nid in active_ids:
                self.propagate_wave(nid, 1.0)

            if resonance < 0.6:
                self.memory_interference.append({
                    'vector': vec.copy(),
                    'text': word,
                    'level': 0,
                    'timestamp': self.t
                })

        total_vec /= len(words)
        self.memory_interference.append({
            'vector': total_vec.copy(),
            'text': ' '.join(words),
            'level': 1,
            'timestamp': self.t
        })

        self.form_hierarchy()

        activation = np.array([n.last_activation for n in self.nodes.values()])
        if len(activation) > 0:
            self.activation_history.append(activation.copy())
            if len(self.activation_history) > 50:
                self.activation_history.pop(0)

        self.t += 1

        if self.t % SLEEP_CYCLE == 0:
            self.start_sleep()

        return f"Sin понял: '{text}'"

    def start_sleep(self):
        self.sleeping = True
        print("\n🌙 Sin засыпает... начинается консолидация памяти.\n")
        threading.Thread(target=self.dream_cycle, daemon=True).start()

    def dream_cycle(self):
        time.sleep(1)
        for _ in range(5):
            if len(self.memory_interference) == 0:
                continue
            mem = random.choice(self.memory_interference)
            vec = mem['vector']
            noise = np.random.normal(0, 0.05, vec.shape)
            dream = vec + noise
            dream /= (np.linalg.norm(dream) + 1e-8)

            sim = self.check_resonance(dream)
            if sim < 0.8:
                self.memory_interference.append({
                    'vector': dream.copy(),
                    'text': f"[сон:{mem['text']}*]",
                    'level': mem['level'],
                    'timestamp': self.t
                })
            time.sleep(0.5)
        self.sleeping = False
        print("\n🧠 Sin проснулся. Память укреплена.\n")

    def respond(self, text):
        if self.sleeping:
            return "Zzz... Sin спит. Но, кажется, что-то бормочет: 'ммм... смысл...'"
        words = self.tokenize(text)
        if not words:
            return "Я слушаю..."

        response_hints = [
            "Это напоминает мне о",
            "Я чувствую сходство с",
            "Это резонирует с идеей",
            "Похоже на",
            "Мне кажется, это связано с"
        ]

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
            hint = random.choice(response_hints)
            return f"{hint} '{best_match}' (схожесть: {best_sim:.2f})."
        elif best_sim > 0.4:
            return f"Частично понимаю. Ещё не до конца ясно ({best_sim:.2f})."
        else:
            if words:
                word = words[0]
                similar_words = self.get_similar_words(word, top_k=3)
                if similar_words:
                    return f"Не знаю '{word}', но знаю: {', '.join(similar_words)}. Расскажи больше?"
            return f"Новое. Ещё не резонирует. Расскажи больше."

    def get_similar_words(self, word, top_k=3):
        try:
            similar = self.embedder.model.most_similar(positive=[word.lower()], topn=top_k)
            return [w for w, s in similar]
        except:
            return []

    def status(self):
        return f"""
        🌐 Sin — Сеть Интуитивного Понимания (Русский)
        Время: {self.t}
        Узлов: {len(self.nodes)}
        Память: {len(self.memory_interference)}
        Состояние: {'Спит' if self.sleeping else 'Бодрствует'}
        Уровни: {len(self.level_nodes[0])} слов, {len(self.level_nodes[1])} фраз
        """

    def visualize_resonance(self):
        if not self.activation_history:
            print("Нет данных для визуализации.")
            return
        plt.figure(figsize=(10, 5))
        data = np.array(self.activation_history)
        if data.size == 0:
            print("Нет активности.")
            return
        plt.imshow(data.T, aspect='auto', cmap='plasma', interpolation='none')
        plt.colorbar(label="Активация")
        plt.title("Волны резонанса в Sin")
        plt.xlabel("Время")
        plt.ylabel("Нейроны")
        plt.tight_layout()
        plt.show()


# === КОНСОЛЬНЫЙ ИНТЕРФЕЙС ===
def main():
    print("🌀 Загрузка Sin — Сеть Интуитивного Понимания (Русский v1.0)")
    try:
        embedder = RuEmbedder(EMBEDDING_FILE)
    except Exception as e:
        print(f"Ошибка: {e}")
        return

    sin = Sin(embedder)

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
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Sin уходит в тишину...")
                break
            elif user_input.lower() == "status":
                print(sin.status())
            elif user_input.lower() == "sleep":
                sin.start_sleep()
            elif user_input.lower() == "visualize":
                sin.visualize_resonance()
            else:
                learn_response = sin.learn(user_input)
                print(learn_response)
                response = sin.respond(user_input)
                print(f"💬 Sin: {response}")
        except KeyboardInterrupt:
            print("\nSin замолкает...")
            break


if __name__ == "__main__":
    main()
