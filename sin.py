import os
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

# ----------------------------------------
# Настройки
# ----------------------------------------
PROJECT_DIR = Path(r"C:\Users\User\Downloads\Sin")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
LOGS_DIR = PROJECT_DIR / "logs"
CONFIG_FILE = PROJECT_DIR / "config.json"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console_width=120, show_path=False)]
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
    "quality_score": 0.0
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
# Генерация "связок" предложений
# ----------------------------------------
def generate_sentence_pairs(n=1000):
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
                a = "Я не понимаю этот вопрос."  # шум
            pairs.append({"instruction": q.strip(), "response": a.strip()})
            progress.update(task, advance=1, description=f"Генерация: {len(pairs)} из {n}")

    # Сохранение
    data_file = DATA_DIR / "pairs.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"[green]Сгенерировано {len(pairs)} пар. Сохранено в {data_file}[/green]")
    return pairs

# ----------------------------------------
# Фильтрация данных (простая валидация)
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
# Оценка качества (упрощённая)
# ----------------------------------------
def evaluate_model(model, eval_pairs, num_samples=10):
    correct_keywords = ["привет", "дела", "анекдот", "пока", "спасибо"]
    score = 0.0
    with Progress(console=console) as progress:
        task = progress.add_task("Оценка качества...", total=num_samples)
        for pair in random.sample(eval_pairs, min(num_samples, len(eval_pairs))):
            q = pair["instruction"]
            inputs = tokenizer(f"Вопрос: {q} Ответ:", return_tensors="pt", truncation=True, max_length=64).to(model.device)
            outputs = model.generate(**inputs, max_length=128)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Простая проверка: содержит ли ответ осмысленные слова
            if len(response) > 10 and any(k in response.lower() for k in correct_keywords):
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
    model.print_trainable_parameters()

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
        push_to_hub=False,
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

    # Сохранение
    model.save_pretrained(MODEL_DIR / "finetuned")
    tokenizer.save_pretrained(MODEL_DIR / "finetuned")
    logger.info("[bold green]Модель сохранена![/bold green]")

    # Оценка
    quality = evaluate_model(model, pairs)
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
# Тест общения
# ----------------------------------------
def test_chat():
    load_model()
    console.print(Panel("ТЕСТ ОБЩЕНИЯ С SIN", style="bold blue"))
    console.print("[yellow]Привет! Я — Sin. Готов к диалогу. Напиши что-нибудь или 'exit' для выхода.[/yellow]")
    while True:
        user_input = console.input("[bold]Вы:[/bold] ")
        if user_input.lower() in ['exit', 'выход', 'quit']:
            break
        prompt = f"Вопрос: {user_input} Ответ:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
        outputs = model.generate(**inputs, max_length=128, temperature=0.7, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Вопрос:" in response:
            response = response.split("Вопрос:")[0].strip()
        console.print(f"[magenta]Sin:[/magenta] {response}")

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
        table.add_row("3", "Тест общения")
        table.add_row("4", "Просмотр статуса")
        table.add_row("5", "Выход")

        console.print(table)
        choice = console.input("[bold]Выберите действие: [/bold]")

        if choice == "1":
            train_model()
        elif choice == "2":
            test_generation()
        elif choice == "3":
            test_chat()
        elif choice == "4":
            status_table = Table(title="Статус Sin", show_header=True)
            status_table.add_column("Параметр")
            status_table.add_column("Значение")
            status_table.add_row("Последнее обучение", config.get("last_trained", "Не было"))
            status_table.add_row("Оценка качества", f"{config.get('quality_score', 0):.2f}")
            status_table.add_row("Модель", str(MODEL_DIR / "finetuned"))
            status_table.add_row("Данные", f"{len(os.listdir(DATA_DIR))} файлов")
            console.print(status_table)
        elif choice == "5":
            console.print("[bold red]До свидания! Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Неверная команда.[/red]")

if __name__ == "__main__":
    main()
