import os
import json
import time
import random
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import faiss
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    pipeline
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
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
LOGS_DIR = PROJECT_DIR / "logs"
CACHE_DIR = PROJECT_DIR / "cache"
CONFIG_FILE = PROJECT_DIR / "config.json"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Логирование с Rich
rich_console = RichConsole(width=120)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=rich_console, show_path=False)]
)
logger = logging.getLogger("Sin")
console = Console()

# ----------------------------------------
# Конфигурация
# ----------------------------------------
default_config = {
    "model_name": "cointegrated/rut5-base",
    "max_length": 128,
    "batch_size": 8,
    "epochs": 3,
    "learning_rate": 2e-4,
    "lora_r": 8,
    "lora_alpha": 32,
    "last_trained": None,
    "quality_score": 0.0,
    "rag_enabled": True,
    "rlhf_enabled": True,
    "history_size": 50  # сколько диалогов хранить
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4, ensure_ascii=False)
    logger.info(f"[green]Конфиг сохранён: {CONFIG_FILE}[/green]")
else:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

# ----------------------------------------
# Загрузка модели и токенизатора
# ----------------------------------------
tokenizer = None
model = None
embedding_model = None

def load_model():
    global tokenizer, model
    model_path = MODEL_DIR / "finetuned"
    if os.path.exists(model_path):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Загрузка модели...", total=None)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        logger.info("[blue]Модель загружена из локальной директории.[/blue]")
    else:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Загрузка базовой модели...", total=None)
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
        logger.info("[yellow]Базовая модель загружена.[/yellow]")

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
            self.emb_pipeline = pipeline("feature-extraction", model="cointegrated/rubert-tiny2", device=0 if torch.cuda.is_available() else -1)
        emb = self.emb_pipeline(text)[0][0]  # [CLS] токен
        return np.array(emb).astype(np.float32)

    def add(self, sentence):
        if len(sentence.strip()) < 3:
            return
        emb = self._get_embedding(sentence).reshape(1, -1)
        if self.embeddings.shape[0] == 0:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack((self.embeddings, emb))
        self.sentences.append(sentence)
        self.index.add(emb)

    def search(self, query, k=3):
        if self.embeddings.shape[0] == 0:
            return []
        q_emb = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(q_emb, k)
        return [self.sentences[i] for i in indices[0] if i < len(self.sentences)]

    def save(self, path):
        path = Path(path)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "sentences.json", "w", encoding="utf-8") as f:
            json.dump(self.sentences, f, ensure_ascii=False, indent=2)

    def load(self, path):
        path = Path(path)
        if not os.path.exists(path / "index.faiss"):
            return
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "sentences.json", "r", encoding="utf-8") as f:
            self.sentences = json.load(f)
        logger.info(f"[blue]RAG-память загружена: {len(self.sentences)} записей[/blue]")

memory = VectorMemory()

# ----------------------------------------
# Генерация "связок" предложений
# ----------------------------------------
def generate_sentence_pairs(n=500):
    templates = [
        ("Привет", "Здравствуй!"),
        ("Как дела?", "У меня всё хорошо, спасибо!"),
        ("Что ты умеешь?", "Я могу отвечать на вопросы и поддерживать беседу."),
        ("Расскажи анекдот", "Почему программисты не ходят в лес? Боятся глубоких рекурсий!"),
        ("Кто ты?", "Я — Sin, ваш искусственный собеседник."),
        ("Погода сегодня", "Сегодня солнечно и тепло."),
        ("Сколько будет 2+2?", "Будет 4."),
        ("Пока", "До встречи!"),
    ]

    pairs = []
    with Progress(
        TextColumn("[blue]Генерация[/blue] [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Генерация пар...", total=n)
        for _ in range(n):
            template = random.choice(templates)
            q = template[0] + " " + " ".join([random.choice(["и?", "скажи", "расскажи", "объясни"]) for _ in range(random.randint(0, 2))])
            a = template[1] + " " + random.choice(["Спрашивай ещё!", "Всё понятно?", ""])
            if random.random() < 0.1:
                a = "Я не понимаю этот вопрос."
            pairs.append({"instruction": q.strip(), "response": a.strip()})
            progress.update(task, advance=1, description=f"Генерация: {len(pairs)} из {n}")

    data_file = DATA_DIR / "pairs.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"[green]Сгенерировано {len(pairs)} пар. Сохранено в {data_file}[/green]")
    return pairs

# ----------------------------------------
# Фильтрация данных
# ----------------------------------------
def filter_invalid_pairs(pairs):
    filtered = []
    invalid = 0
    with Progress(
        TextColumn("[red]Фильтрация[/red] [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Фильтрация...", total=len(pairs))
        for pair in pairs:
            q, a = pair["instruction"].strip(), pair["response"].strip()
            if len(q) < 3 or len(a) < 3:
                invalid += 1
                continue
            if any(bad in q.lower() for bad in ["xxx", "porn", "fuck"]):
                invalid += 1
                continue
            filtered.append(pair)
            progress.update(task, advance=1)
    logger.info(f"[yellow]Отфильтровано {invalid} невалидных пар. Осталось: {len(filtered)}[/yellow]")
    return filtered

# ----------------------------------------
# Подготовка данных
# ----------------------------------------
def load_or_generate_data():
    pairs_file = DATA_DIR / "pairs.json"
    if os.path.exists(pairs_file):
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        logger.info(f"[blue]Загружено {len(pairs)} пар из файла.[/blue]")
    else:
        logger.info("[yellow]Файл пар не найден. Генерация...[/yellow]")
        pairs = generate_sentence_pairs(n=500)
    return filter_invalid_pairs(pairs)

# ----------------------------------------
# Форматирование под обучение
# ----------------------------------------
def format_dataset(pairs):
    texts = [f"Вопрос: {p['instruction']} Ответ: {p['response']}" for p in pairs]
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=config["max_length"]),
        batched=True,
        remove_columns=["text"]
    )

# ----------------------------------------
# RLHF: Человеческая оценка
# ----------------------------------------
def collect_human_feedback(pairs, n=5):
    feedback_data = []
    console.print(Panel("🧠 RLHF: Оцените ответы Sin", style="bold yellow"))
    load_model()
    for pair in random.sample(pairs, min(n, len(pairs))):
        q = pair["instruction"]
        inputs = tokenizer(f"Вопрос: {q} Ответ:", return_tensors="pt", truncation=True, max_length=64).to(model.device)
        outputs = model.generate(**inputs, max_length=128)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("Вопрос: " + q + " Ответ:", "").strip()

        console.print(f"[cyan]Вопрос:[/cyan] {q}")
        console.print(f"[magenta]Sin:[/magenta] {response}")
        rating = console.input("Оценка (1-5, или пропустить): ").strip()
        if rating in ["1", "2", "3", "4", "5"]:
            feedback_data.append({
                "input": q,
                "output": response,
                "rating": int(rating)
            })
    # Сохраняем фидбэк
    fb_file = DATA_DIR / "feedback.json"
    old = []
    if os.path.exists(fb_file):
        with open(fb_file, 'r', encoding='utf-8') as f:
            old = json.load(f)
    old.extend(feedback_data)
    with open(fb_file, 'w', encoding='utf-8') as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    logger.info(f"[green]Собрано {len(feedback_data)} оценок. Сохранено.[/green]")

# ----------------------------------------
# Оценка качества
# ----------------------------------------
def evaluate_model(eval_pairs, num_samples=10):
    keywords = ["привет", "дела", "анекдот", "пока", "спасибо", "понятно"]
    score = 0.0
    with Progress(console=console) as progress:
        task = progress.add_task("Оценка качества...", total=num_samples)
        for pair in random.sample(eval_pairs, min(num_samples, len(eval_pairs))):
            q = pair["instruction"]
            inputs = tokenizer(f"Вопрос: {q} Ответ:", return_tensors="pt", truncation=True, max_length=64).to(model.device)
            outputs = model.generate(**inputs, max_length=128)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if len(response) > 10 and any(k in response.lower() for k in keywords):
                score += 1
            progress.update(task, advance=1)
    return score / num_samples

# ----------------------------------------
# Обучение
# ----------------------------------------
def train_model():
    global model, tokenizer
    load_model()

    pairs = load_or_generate_data()
    dataset = format_dataset(pairs)

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR / "finetuned",
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        save_steps=100,
        logging_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        fp16=True,
        report_to=None,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        logging_dir=LOGS_DIR,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Обучение Sin...", total=100)
        logger.info("[bold green]Начинается обучение...[/bold green]")
        trainer.train()
        progress.update(task, completed=100)

    model.save_pretrained(MODEL_DIR / "finetuned")
    tokenizer.save_pretrained(MODEL_DIR / "finetuned")
    logger.info("[bold green]Модель сохранена![/bold green]")

    quality = evaluate_model(pairs)
    config["quality_score"] = quality
    config["last_trained"] = datetime.now().isoformat()
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logger.info(f"[blue]Оценка качества: {quality:.2f}[/blue]")

# ----------------------------------------
# Тест генерации
# ----------------------------------------
def test_generation():
    load_model()
    console.print(Panel("ТЕСТ ГЕНЕРАЦИИ ТЕКСТА", style="bold magenta"))
    while True:
        prompt = console.input("[bold cyan]Введите текст (или 'exit'): [/bold cyan]")
        if prompt.lower() == 'exit':
            break
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
        outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.8)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        console.print(f"[green]Sin:[/green] {response}")

# ----------------------------------------
# Тест общения (с RAG)
# ----------------------------------------
def test_chat():
    load_model()
    memory.load(CACHE_DIR)
    history = []
    console.print(Panel("ТЕСТ ОБЩЕНИЯ С SIN (с памятью)", style="bold blue"))
    console.print("[yellow]Привет! Я — Sin. Я помню наш диалог. Напиши 'exit' для выхода.[/yellow]")

    while True:
        user_input = console.input("[bold]Вы:[/bold] ")
        if user_input.lower() in ['exit', 'выход', 'quit']:
            break

        # RAG: поиск похожих вопросов
        relevant = memory.search(user_input, k=2)
        context = "\n".join([f"Ранее: {r}" for r in relevant]) if relevant else ""

        # Добавляем в историю
        history.append(f"Вы: {user_input}")
        memory.add(user_input)

        # Формируем промпт
        prompt = (context + "\n" if context else "") + f"Вопрос: {user_input} Ответ:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(model.device)
        outputs = model.generate(**inputs, max_length=128, temperature=0.7, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        console.print(f"[magenta]Sin:[/magenta] {response}")
        history.append(f"Sin: {response}")
        memory.add(response)

        # Ограничиваем историю
        if len(history) > config["history_size"]:
            history.pop(0)

    # Сохраняем память
    memory.save(CACHE_DIR)

# ----------------------------------------
# Главное меню
# ----------------------------------------
def main():
    console.print(Panel("🤖 Добро пожаловать в систему обучения Sin!", style="bold green"))

    while True:
        table = Table(title="Меню Sin", show_header=True, header_style="bold magenta")
        table.add_column("Команда", style="dim")
        table.add_column("Описание")
        table.add_row("1", "Автообучение (генерация + обучение)")
        table.add_row("2", "Тест генерации текста")
        table.add_row("3", "Тест общения (с памятью)")
        table.add_row("4", "Сбор RLHF-оценок (человеческая оценка)")
        table.add_row("5", "Просмотр статуса")
        table.add_row("6", "Выход")

        console.print(table)
        choice = console.input("[bold]Выберите действие: [/bold]")

        if choice == "1":
            train_model()
        elif choice == "2":
            test_generation()
        elif choice == "3":
            test_chat()
        elif choice == "4":
            pairs = load_or_generate_data()
            collect_human_feedback(pairs, n=5)
        elif choice == "5":
            status_table = Table(title="Статус Sin", show_header=True)
            status_table.add_column("Параметр")
            status_table.add_column("Значение")
            status_table.add_row("Последнее обучение", config.get("last_trained", "Не было"))
            status_table.add_row("Оценка качества", f"{config.get('quality_score', 0):.2f}")
            status_table.add_row("RLHF оценок", str(len([f for f in os.listdir(DATA_DIR) if 'feedback' in f])))
            status_table.add_row("RAG записей", str(len(memory.sentences) if hasattr(memory, 'sentences') else 0))
            status_table.add_row("Модель", str(MODEL_DIR / "finetuned"))
            console.print(status_table)
        elif choice == "6":
            console.print("[bold red]До свидания! Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Неверная команда.[/red]")

if __name__ == "__main__":
    main()
