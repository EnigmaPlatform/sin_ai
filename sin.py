import os
import json
import time
import random
import logging
from pathlib import Path

import torch
import faiss
import numpy as np
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

# ----------------------------------------
# Настройки
# ----------------------------------------
PROJECT_DIR = Path(r"C:\Users\User\Downloads\Sin")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
CACHE_DIR = PROJECT_DIR / "cache"
CONFIG_FILE = PROJECT_DIR / "config.json"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Логирование
rich_console = RichConsole(width=120)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=rich_console, show_path=False)]
)
logger = logging.getLogger("Sin")
console = Console()

# Конфиг
config = {
    "model_name": "IlyaGusev/saiga_llama3_8b",
    "max_length": 1024,
    "lora_r": 64,
    "lora_alpha": 16,
    "batch_size": 1,
    "last_trained": None,
    "quality_score": 0.0
}

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config.update(json.load(f))
else:
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# ----------------------------------------
# Скачивание всей модели
# ----------------------------------------
def download_model():
    if not (MODEL_DIR / "config.json").exists():
        console.print("[bold]Скачивание модели...[/bold]")
        try:
            snapshot_download(
                repo_id=config["model_name"],
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
                max_workers=2
            )
            logger.info("[green]Модель успешно скачана.[/green]")
        except Exception as e:
            logger.error(f"[red]Ошибка скачивания: {e}[/red]")
            return False
    else:
        logger.info("[blue]Модель уже загружена локально.[/blue]")
    return True

# ----------------------------------------
# Загрузка модели
# ----------------------------------------
model = None
tokenizer = None

def load_model_base():
    global model, tokenizer
    if not download_model():
        raise RuntimeError("Не удалось скачать модель")

    if model is not None:
        return model, tokenizer

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_DIR,
            max_seq_length=config["max_length"],
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_r"],
            alpha=config["lora_alpha"],
            dropout=0.05,
            bias="none",
            random_state=3407
        )
        logger.info("[green]Модель и токенизатор загружены.[/green]")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки: {e}")

# ----------------------------------------
# Векторная память (RAG)
# ----------------------------------------
class VectorMemory:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.sentences = []
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.emb_pipeline = None

    def _get_embedding(self, text):
        if self.emb_pipeline is None:
            from transformers import pipeline
            self.emb_pipeline = pipeline("feature-extraction", model="cointegrated/rubert-tiny2")
        try:
            return np.array(self.emb_pipeline(text)[0][0]).astype(np.float32)
        except:
            return np.random.rand(self.dim).astype(np.float32)

    def add(self, sentence):
        if len(sentence.strip()) < 3:
            return
        emb = self._get_embedding(sentence).reshape(1, -1)
        self.embeddings = np.vstack((self.embeddings, emb)) if self.embeddings.shape[0] > 0 else emb
        self.sentences.append(sentence)
        self.index.add(emb)

    def search(self, query, k=3):
        if self.embeddings.shape[0] == 0:
            return []
        q_emb = self._get_embedding(query).reshape(1, -1)
        _, indices = self.index.search(q_emb, k)
        return [self.sentences[i] for i in indices[0] if i < len(self.sentences)]

    def save(self):
        import numpy as np
        faiss.write_index(self.index, str(CACHE_DIR / "index.faiss"))
        np.save(CACHE_DIR / "embeddings.npy", self.embeddings)
        with open(CACHE_DIR / "sentences.json", "w", encoding="utf-8") as f:
            json.dump(self.sentences, f, ensure_ascii=False, indent=2)
        logger.info(f"[blue]Память сохранена: {len(self.sentences)} записей[/blue]")

    def load(self):
        if (CACHE_DIR / "index.faiss").exists():
            self.index = faiss.read_index(str(CACHE_DIR / "index.faiss"))
            self.embeddings = np.load(CACHE_DIR / "embeddings.npy")
            with open(CACHE_DIR / "sentences.json", "r", encoding="utf-8") as f:
                self.sentences = json.load(f)
            logger.info(f"[blue]Память загружена: {len(self.sentences)} записей[/blue]")

memory = VectorMemory()
memory.load()

# ----------------------------------------
# Сбор RLHF-оценок
# ----------------------------------------
def collect_human_feedback():
    global model, tokenizer
    model, tokenizer = load_model_base()
    feedback_file = DATA_DIR / "feedback.jsonl"

    console.print(Panel("🧠 Оцените ответы Sin (1–5)", style="bold yellow"))
    feedback = []

    prompts = ["Привет", "Как дела?", "Расскажи анекдот", "Кто ты?", "Погода", "2+2", "Пока", "Что умеешь?"]

    for prompt in random.sample(prompts, 3):
        prompt_text = f"<|user|>\n{prompt}<|bot|>\n"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        try:
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.8, top_p=0.9)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|bot|>")[-1].strip().split("<|")[0]
        except:
            response = "Не могу ответить."

        console.print(f"[cyan]Вы:[/cyan] {prompt}")
        console.print(f"[magenta]Sin:[/magenta] {response}")
        rating = console.input("Оценка (1-5): ").strip()
        if rating in "12345":
            feedback.append({"input": prompt_text.strip(), "output": response, "score": int(rating)})

    with open(feedback_file, "a", encoding="utf-8") as f:
        for item in feedback:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"[green]Сохранено {len(feedback)} оценок.[/green]")

# ----------------------------------------
# Тест общения
# ----------------------------------------
def test_chat():
    global model, tokenizer
    model, tokenizer = load_model_base()
    console.print(Panel("💬 Sin — режим общения", style="bold blue"))
    console.print("[yellow]Напишите 'выход' для завершения.[/yellow]")

    while True:
        user_input = console.input("[bold]Вы:[/bold] ").strip()
        if user_input.lower() in ['выход', 'exit']:
            break

        ctx = "\n".join([f"Ранее: {x}" for x in memory.search(user_input, k=2)])
        prompt = f"{ctx}\n<|user|>\n{user_input}<|bot|>\n" if ctx else f"<|user|>\n{user_input}<|bot|>\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        try:
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, top_p=0.9)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|bot|>")[-1].strip().split("<|")[0]
        except:
            response = "Извини, не могу ответить."

        console.print(f"[magenta]Sin:[/magenta] {response}")

        memory.add(user_input)
        memory.add(response)

    memory.save()

# ----------------------------------------
# Главное меню
# ----------------------------------------
def main():
    console.print(Panel("🤖 Sin — ваш ИИ-ассистент", style="bold green"))
    while True:
        table = Table(title="Меню", show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim")
        table.add_column("Действие")
        table.add_row("1", "Собрать RLHF оценки")
        table.add_row("2", "Тест общения")
        table.add_row("3", "Статус")
        table.add_row("4", "Выход")
        console.print(table)

        choice = console.input("[bold]Выберите: [/bold]")
        if choice == "1":
            collect_human_feedback()
        elif choice == "2":
            test_chat()
        elif choice == "3":
            fb_file = DATA_DIR / "feedback.jsonl"
            n_fb = sum(1 for _ in open(fb_file, 'r', encoding='utf-8') if _.strip()) if fb_file.exists() else 0
            status = Table(title="Статус")
            status.add_row("Последнее обучение", config.get("last_trained", "—"))
            status.add_row("Оценок RLHF", str(n_fb))
            status.add_row("RAG записей", str(len(memory.sentences)))
            console.print(status)
        elif choice == "4":
            console.print("[bold red]До свидания, Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Ошибка.[/red]")

if __name__ == "__main__":
    main()
