import os
import json
import time
import random
import logging
from pathlib import Path

import torch
import faiss
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download, HfApi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

# ----------------------------------------
# Настройки
# ----------------------------------------
PROJECT_DIR = Path(r"C:\Users\alex\Downloads\Sin")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
CACHE_DIR = PROJECT_DIR / "cache"
CONFIG_FILE = PROJECT_DIR / "config.json"
LOGS_DIR = PROJECT_DIR / "logs"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

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
    "model_name": "cointegrated/rut5-base",  # ✅ Правильное имя
    "max_length": 512,
    "lora_r": 8,
    "lora_alpha": 32,
    "batch_size": 4,
    "epochs": 1,
    "learning_rate": 2e-4,
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
# Проверка подключения к HF
# ----------------------------------------
def is_online():
    try:
        api = HfApi()
        api.list_models(limit=1)
        return True
    except:
        return False

# ----------------------------------------
# Скачивание модели
# ----------------------------------------
def download_model():
    if not (MODEL_DIR / "config.json").exists():
        if not is_online():
            logger.warning("[yellow]Нет интернета. Работаю в оффлайн-режиме.[/yellow]")
            return False

        console.print("[bold]Скачивание модели cointegrated/rut5-base...[/bold]")
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
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return tokenizer, model

    # Попробуем загрузить локально
    if (MODEL_DIR / "config.json").exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            logger.info("[green]Модель загружена локально.[/green]")
            return tokenizer, model
        except Exception as e:
            logger.error(f"[red]Ошибка загрузки локальной модели: {e}[/red]")

    # Если не получилось — скачиваем
    if download_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            logger.info("[green]Модель загружена после скачивания.[/green]")
            return tokenizer, model
        except Exception as e:
            logger.error(f"[red]Ошибка загрузки после скачивания: {e}[/red]")

    # Фолбэк: используем встроенные данные
    logger.warning("[yellow]Использую режим без модели (тестовые ответы).[/yellow]")
    return None, None

# ----------------------------------------
# Векторная память (RAG)
# ----------------------------------------
class VectorMemory:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.sentences = []
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.emb_pipeline = None

    def _get_embedding(self, text):
        if self.emb_pipeline is None:
            try:
                self.emb_pipeline = pipeline("feature-extraction", model="cointegrated/rubert-tiny2")
            except:
                return np.random.rand(self.dim).astype(np.float32)
        try:
            emb = self.emb_pipeline(text)[0][0]
            return np.array(emb).astype(np.float32)
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
# Генерация данных
# ----------------------------------------
def generate_data():
    return [
        ("Привет", "Здравствуй!"),
        ("Как дела?", "Хорошо, спасибо!"),
        ("Расскажи анекдот", "Не программисты ли ходят в лес?"),
        ("Кто ты?", "Я — Sin, твой помощник."),
        ("Погода", "Солнечно."),
        ("2+2", "4"),
        ("Пока", "До встречи!")
    ]

# ----------------------------------------
# Сбор RLHF-оценок
# ----------------------------------------
def collect_human_feedback():
    feedback_file = DATA_DIR / "feedback.jsonl"
    tokenizer, model = load_model()

    console.print(Panel("🧠 Оцените ответы Sin (1–5)", style="bold yellow"))
    feedback = []

    for q, a in generate_data():
        if model is not None:
            prompt = f"Вопрос: {q} Ответ:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            try:
                outputs = model.generate(**inputs, max_length=300)
                a = tokenizer.decode(outputs[0], skip_special_tokens=True)
            except:
                pass

        console.print(f"[cyan]Вопрос:[/cyan] {q}")
        console.print(f"[magenta]Sin:[/magenta] {a}")
        rating = console.input("Оценка (1-5): ").strip()
        if rating in "12345":
            feedback.append({"input": q, "output": a, "score": int(rating)})

    if feedback:
        mode = "a" if feedback_file.exists() else "w"
        with open(feedback_file, mode, encoding="utf-8") as f:
            for item in feedback:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"[green]Сохранено {len(feedback)} оценок.[/green]")

# ----------------------------------------
# Обучение
# ----------------------------------------
def train_model():
    tokenizer, model = load_model()
    if model is None:
        logger.warning("[yellow]Нет модели для обучения.[/yellow]")
        return

    data = generate_data()
    texts = [f"Вопрос: {q} Ответ: {a}" for q, a in data]
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})

    model = get_peft_model(model, LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    ))

    from transformers import TrainingArguments, Trainer
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=256)),
        tokenizer=tokenizer
    )

    with Progress(SpinnerColumn(), TextColumn("Обучение..."), BarColumn(), console=console) as progress:
        progress.add_task("", total=None)
        trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    config["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logger.info("[bold green]Обучение завершено.[/bold green]")

# ----------------------------------------
# Тест общения
# ----------------------------------------
def test_chat():
    tokenizer, model = load_model()
    console.print(Panel("💬 Sin — режим общения", style="bold blue"))
    console.print("[yellow]Напишите 'выход' для завершения.[/yellow]")

    while True:
        user_input = console.input("[bold]Вы:[/bold] ").strip()
        if user_input.lower() in ['выход', 'exit']:
            break

        # RAG
        ctx = "\n".join([f"Ранее: {x}" for x in memory.search(user_input, k=2)])
        prompt = f"{ctx}\nВопрос: {user_input} Ответ:" if ctx else f"Вопрос: {user_input} Ответ:"

        response = "Я не понял."
        if model is not None:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                outputs = model.generate(**inputs, max_length=300)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("Ответ:")[-1].strip()
            except:
                pass

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
        table.add_row("2", "Обучить модель")
        table.add_row("3", "Тест общения")
        table.add_row("4", "Статус")
        table.add_row("5", "Выход")
        console.print(table)

        choice = console.input("[bold]Выберите: [/bold]")
        if choice == "1": collect_human_feedback()
        elif choice == "2": train_model()
        elif choice == "3": test_chat()
        elif choice == "4":
            fb_file = DATA_DIR / "feedback.jsonl"
            n_fb = sum(1 for _ in open(fb_file, 'r', encoding='utf-8') if _.strip()) if fb_file.exists() else 0
            status = Table(title="Статус")
            status.add_row("Последнее обучение", config.get("last_trained", "—"))
            status.add_row("Оценок RLHF", str(n_fb))
            status.add_row("RAG записей", str(len(memory.sentences)))
            console.print(status)
        elif choice == "5":
            console.print("[bold red]До свидания, Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Ошибка.[/red]")

if __name__ == "__main__":
    main()
