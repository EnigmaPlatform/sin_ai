import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import deque
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    SFTTrainer  # Импорт SFTTrainer
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
import logging

# ----------------------------------------
# Настройки: папка Sin в директории запуска
# ----------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BASE_DIR / "Sin"
MODEL_DIR = PROJECT_DIR / "model"
DATA_DIR = PROJECT_DIR / "data"
MEMORY_DIR = PROJECT_DIR / "memory"
CONFIG_FILE = PROJECT_DIR / "config.json"
LOGS_DIR = PROJECT_DIR / "logs"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Логирование
rich_console = Console(width=120)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=rich_console, show_path=False)]
)
logger = logging.getLogger("Sin")

console = Console()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Конфигурация ---
EMOTION_ENGINE_CONFIG = {
    "base_emotions": {
        "happy": {"icon": "😊", "triggers": ["рад", "счастлив", "люблю"]},
        "sad": {"icon": "😢", "triggers": ["грустн", "печаль", "плак"]},
        "angry": {"icon": "😠", "triggers": ["злюсь", "бесит", "ненавижу"]},
        "fear": {"icon": "😨", "triggers": ["боюсь", "страх", "пугает"]},
        "surprise": {"icon": "😲", "triggers": ["невероятно", "удивитель"]},
        "disgust": {"icon": "🤢", "triggers": ["отврат", "мерзк", "противно"]},
        "neutral": {"icon": "😐", "triggers": []}
    },
    "decay_rate": 0.95,
    "intensity_threshold": 0.3,
    "max_memory": 1000
}

# ----------------------------------------
# Загрузка модели с оптимизацией для CPU
# ----------------------------------------
def load_optimized_model():
    local = (MODEL_DIR / "config.json").exists()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR if local else "cointegrated/rut5-base",
            local_files_only=local,
            use_fast=False  # Отключаем fast-токенизатор
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR if local else "cointegrated/rut5-base",
            local_files_only=local,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        console.print("[green]Модель успешно загружена[/green]")
        return tokenizer, model
    except Exception as e:
        console.print(f"[red]Ошибка загрузки модели: {e}[/red]")
        return None, None

# --- Класс эмоционального состояния ---
class EmotionalState:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.5
        self.emotion_history = deque(maxlen=50)
        self.long_term_mood = 0.5
        self.triggers_activated = set()

    def update(self, text: str):
        text_lower = text.lower()
        for emotion, data in EMOTION_ENGINE_CONFIG["base_emotions"].items():
            for trigger in data["triggers"]:
                if trigger in text_lower:
                    self.current_emotion = emotion
                    self.emotion_intensity = min(1.0, self.emotion_intensity + 0.3)
                    self.triggers_activated.add(emotion)

        self.emotion_intensity *= EMOTION_ENGINE_CONFIG["decay_rate"]
        if self.emotion_intensity < EMOTION_ENGINE_CONFIG["intensity_threshold"]:
            self.current_emotion = "neutral"

        sentiment = self.analyze_sentiment(text)
        self.long_term_mood = 0.9 * self.long_term_mood + 0.1 * sentiment

        self.emotion_history.append({
            "timestamp": datetime.now().isoformat(),
            "emotion": self.current_emotion,
            "intensity": self.emotion_intensity,
            "text": text
        })

    def analyze_sentiment(self, text: str) -> float:
        positive_words = ["хорош", "прекрасн", "рад", "счастлив"]
        negative_words = ["плох", "ужасн", "грустн", "злюсь"]
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        return (pos_count - neg_count) / max(1, pos_count + neg_count)

    def get_state(self) -> Dict:
        return {
            "current_emotion": self.current_emotion,
            "emotion_icon": EMOTION_ENGINE_CONFIG["base_emotions"][self.current_emotion]["icon"],
            "intensity": self.emotion_intensity,
            "long_term_mood": self.long_term_mood,
            "triggers": list(self.triggers_activated)
        }

# --- Класс долговременной памяти ---
class LongTermMemory:
    def __init__(self):
        self.memory_file = MEMORY_DIR / "long_term_memory.json"
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.memories = []
        if self.memory_file.exists():
            self.load_memory()

    def add_memory(self, text: str, emotion_state: Dict):
        embedding = self.sentence_model.encode(text)
        memory = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion_state,
            "embedding": embedding.tolist()
        }
        self.memories.append(memory)
        if len(self.memories) > EMOTION_ENGINE_CONFIG["max_memory"]:
            self.memories.pop(0)
        self.save_memory()

    def find_related_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.memories:
            return []
        query_embedding = self.sentence_model.encode(query)
        similarities = cosine_similarity([query_embedding], [np.array(m["embedding"]) for m in self.memories])[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memories[i] for i in top_indices]

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump({"memories": self.memories}, f, ensure_ascii=False, indent=2)

    def load_memory(self):
        with open(self.memory_file, "r", encoding="utf-8") as f:
            self.memories = json.load(f).get("memories", [])

# --- Основной класс бота ---
class EmotionalChatBot:
    def __init__(self):
        self.tokenizer, self.model = load_optimized_model()
        self.emotion_engine = EmotionalState()
        self.memory = LongTermMemory()
        self.conversation_history = []
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "personality_traits": {"openness": 0.7, "agreeableness": 0.8},
                "last_trained": None
            }

    def generate_response(self, user_input: str) -> str:
        self.emotion_engine.update(user_input)
        emotion_state = self.emotion_engine.get_state()
        self.memory.add_memory(user_input, emotion_state)

        related = self.memory.find_related_memories(user_input, top_k=2)
        context = "\n".join([f"Ранее: {m['text']}" for m in related])

        prompt = f"""
        Ты — Sin, дружелюбный и чуткий собеседник.
        {context}
        Эмоция: {emotion_state['emotion_icon']} ({emotion_state['current_emotion']}, {emotion_state['intensity']:.2f})
        Пользователь: {user_input}
        Твой ответ:
        """.strip()

        if self.model is None:
            return "Я не могу отвечать (модель не загружена)"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7 + emotion_state["intensity"] * 0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            if not response:
                response = "Я понял тебя."
            return f"{emotion_state['emotion_icon']} {response}"
        except Exception as e:
            return f"Ошибка: {str(e)}"

    def save_state(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

# ----------------------------------------
# SFT: Обучение на парах вопрос-ответ
# ----------------------------------------
def train_sft(bot: EmotionalChatBot):
    if bot.model is None:
        logger.warning("[yellow]Нет модели для обучения.[/yellow]")
        return

    from datasets import Dataset
    # Пример данных
    data = [
        {"text": "Привет Ответ: Здравствуй! Как дела?"},
        {"text": "Как дела? Ответ: У меня всё хорошо, спасибо!"},
        {"text": "Расскажи анекдот Ответ: Почему программисты не ходят в лес? Боятся рекурсии!"},
    ]
    dataset = Dataset.from_list(data)

    # LoRA
    model = get_peft_model(bot.model, LoraConfig(r=8, lora_alpha=32, target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"))

    # Тренировка
    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=LOGS_DIR,
            per_device_train_batch_size=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=None
        ),
        train_dataset=dataset,
        tokenizer=bot.tokenizer,
        max_seq_length=256,
        dataset_text_field="text"
    )

    with Progress(SpinnerColumn(), TextColumn("SFT обучение..."), BarColumn(), console=console) as progress:
        progress.add_task("", total=None)
        trainer.train()

    model.save_pretrained(MODEL_DIR)
    bot.tokenizer.save_pretrained(MODEL_DIR)
    bot.config["last_trained"] = datetime.now().isoformat()
    bot.save_state()
    logger.info("[bold green]SFT обучение завершено.[/bold green]")

# ----------------------------------------
# Главное меню
# ----------------------------------------
def main():
    console.print(Panel.fit("🤖 [bold green]Sin — ваш эмоциональный ассистент[/bold green]"))
    console.print("Напишите 'выход' для завершения диалога\n")

    bot = EmotionalChatBot()

    while True:
        table = Table(title="Меню", show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim")
        table.add_column("Действие")
        table.add_row("1", "Поговорить с Sin")
        table.add_row("2", "Собрать RLHF оценки")
        table.add_row("3", "SFT: дообучить модель")
        table.add_row("4", "Выход")
        console.print(table)

        choice = console.input("[bold]Выберите: [/bold]")
        if choice == "1":
            while True:
                user_input = console.input("[bold blue]Вы:[/bold blue] ").strip()
                if user_input.lower() in ("выход", "exit", "quit"):
                    break
                response = bot.generate_response(user_input)
                console.print(f"[bold magenta]Sin:[/bold magenta] {response}")
                emotion_state = bot.emotion_engine.get_state()
                table = Table(title="Состояние", show_header=False)
                table.add_row("Эмоция", f"{emotion_state['emotion_icon']} {emotion_state['current_emotion']}")
                table.add_row("Интенсивность", f"{emotion_state['intensity']:.2f}")
                table.add_row("Настроение", "😊 Хорошее" if emotion_state['long_term_mood'] > 0.3 else "😐 Нейтральное" if -0.3 <= emotion_state['long_term_mood'] <= 0.3 else "😢 Плохое")
                console.print(table)
        elif choice == "2":
            collect_rlhf_feedback(bot)
        elif choice == "3":
            train_sft(bot)
        elif choice == "4":
            bot.save_state()
            console.print("[bold red]До свидания, Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Ошибка.[/red]")

if __name__ == "__main__":
    main()
