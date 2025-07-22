import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from collections import deque
import gradio as gr
from vosk import Model, KaldiRecognizer
import pyaudio
from ctransformers import AutoModelForCausalLM
from whoosh.index import create_in, open_dir
from whoosh.fields import *
import os
import threading

# --- Конфигурация ---
MODEL_NAME = "saiga"  # Лёгкая русскоязычная модель (4B)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_HISTORY = 5  # Глубина контекста
VOSK_MODEL_PATH = "vosk-model-small-ru-0.22"  # Скачать с alphacephei.com/vosk/models
WHOOSH_INDEX_DIR = "whoosh_index"

# --- Инициализация Whoosh ---
def setup_whoosh():
    if not os.path.exists(WHOOSH_INDEX_DIR):
        os.mkdir(WHOOSH_INDEX_DIR)
        schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
        ix = create_in(WHOOSH_INDEX_DIR, schema)
    else:
        ix = open_dir(WHOOSH_INDEX_DIR)
    return ix

# --- Класс ассистента ---
class Assistant:
    def __init__(self):
        # Память диалога
        self.history = deque(maxlen=MAX_HISTORY)
        
        # База знаний (RAG)
        self.client = chromadb.PersistentClient(path="db/")
        self.collection = self.client.get_or_create_collection(name="knowledge")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Whoosh поиск
        self.whoosh_index = setup_whoosh()
        
        # Ускоренная модель через ctransformers
        self.llm = AutoModelForCausalLM.from_pretrained(
            "saiga-llama3-8b.gguf",
            model_type="llama",
            gpu_layers=1 if torch.cuda.is_available() else 0
        )
        
        # Эмоции
        self.mood = 50  # 0-100
        self.recognizer = None
        self.init_voice_recognition()
        
    def init_voice_recognition(self):
        """Инициализация голосового ввода"""
        if os.path.exists(VOSK_MODEL_PATH):
            model = Model(VOSK_MODEL_PATH)
            self.recognizer = KaldiRecognizer(model, 16000)
            self.mic = pyaudio.PyAudio()
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8192
            )
    
    def listen_voice(self):
        """Запись голоса и преобразование в текст"""
        if not self.recognizer:
            return ""
        
        print("Говорите...")
        self.stream.start_stream()
        while True:
            data = self.stream.read(4096)
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                return json.loads(result)["text"]
    
    def update_mood(self, text):
        """Динамическое обновление настроения"""
        text_lower = text.lower()
        if "спасибо" in text_lower or "thanks" in text_lower:
            self.mood += 10
        elif "тупой" in text_lower or "stupid" in text_lower:
            self.mood -= 20
        self.mood = max(0, min(100, self.mood))
        
    def add_to_knowledge(self, text, title=""):
        """Добавление текста в базу знаний"""
        # Для векторного поиска
        embeddings = self.embedding_model.encode(text)
        self.collection.add(
            embeddings=[embeddings.tolist()],
            documents=[text],
            ids=[f"doc_{len(self.collection.get()['ids'])}"]
        )
        
        # Для полнотекстового поиска
        writer = self.whoosh_index.writer()
        writer.add_document(title=title, content=text)
        writer.commit()
    
    def search_knowledge(self, query):
        """Поиск в локальной базе"""
        # Векторный поиск
        query_embedding = self.embedding_model.encode(query).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        # Полнотекстовый поиск
        with self.whoosh_index.searcher() as searcher:
            text_results = searcher.find("content", query)
        
        # Комбинируем результаты
        best_match = ""
        if vector_results['documents']:
            best_match += f"[Векторный поиск]: {vector_results['documents'][0][0]}\n"
        if text_results:
            best_match += f"[Текстовый поиск]: {text_results[0]['content']}"
        
        return best_match if best_match else None
    
    def generate_response(self, user_input):
        """Генерация ответа с учётом контекста"""
        # Поиск в знаниях
        knowledge = self.search_knowledge(user_input)
        context = f"Контекст: {knowledge}" if knowledge else ""
        
        # Формирование промта
        mood_status = "😊" if self.mood > 60 else "😐" if self.mood > 30 else "😒"
        prompt = f"""
        Ты — Алекс, IT-ассистент. Настроение: {mood_status}.
        {context}
        История: {list(self.history)}
        Пользователь: {user_input}
        Ответ:
        """
        
        # Ускоренная генерация через ctransformers
        response = self.llm(
            prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        # Обновление истории
        self.history.append({"user": user_input, "assistant": response})
        self.update_mood(user_input)
        
        return response

# --- Графический интерфейс ---
assistant = Assistant()

def chat(message, history):
    response = assistant.generate_response(message)
    return response

def voice_input():
    text = assistant.listen_voice()
    return text

with gr.Blocks() as demo:
    with gr.Tab("Текстовый чат"):
        gr.ChatInterface(fn=chat)
    
    with gr.Tab("Голосовой ввод"):
        voice_recording = gr.Button("Запись голоса")
        voice_text = gr.Textbox(label="Распознанный текст")
        voice_recording.click(fn=voice_input, outputs=voice_text)

# --- Инициализация данных ---
if not os.listdir(WHOOSH_INDEX_DIR):
    assistant.add_to_knowledge(
        "Python — интерпретируемый язык программирования",
        "Python"
    )
    assistant.add_to_knowledge(
        "Москва — столица России", 
        "Москва"
    )

# --- Запуск ---
if __name__ == "__main__":
    demo.launch()
