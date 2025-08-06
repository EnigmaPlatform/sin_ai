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
    AutoTokenizer,
    pipeline
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch
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

# ✅ Исправление: RichHandler без console_width
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
    "model_name": "sberbank-ai/rugpt3small",  # causal LM для PPO
    "max_length": 128,
    "batch_size": 2,
    "epochs": 3,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "last_trained": None,
    "quality_score": 0.0,
    "history_size": 50
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4, ensure_ascii=False)
else:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

# ----------------------------------------
# Глобальные переменные
# ----------------------------------------
tokenizer = None
model = None

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
        try:
            emb = self.emb_pipeline(text)[0][0]
            return np.array(emb).astype(np.float32)
        except Exception as e:
            return np.random.rand(self.dim).astype(np.float32)

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
        _, indices = self.index.search(q_emb, k)
        return [self.sentences[i] for i in indices[0] if i < len(self.sentences)]

    def save(self, path):
        path = Path(path)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "sentences.json", "w", encoding="utf-8") as f:
            json.dump(self.sentences, f, ensure_ascii=False, indent=2)

    def load(self, path):
        path = Path(path)
        if not (path / "index.faiss").exists():
            return
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "sentences.json", "r", encoding="utf-8") as f:
            self.sentences = json.load(f)
        logger.info(f"[blue]RAG: загружено {len(self.sentences)} записей[/blue]")

memory = VectorMemory()

# ----------------------------------------
# Сбор RLHF-оценок
# ----------------------------------------
def collect_human_feedback(n=5):
    feedback_file = DATA_DIR / "feedback.jsonl"
    load_model_base()

    console.print(Panel("🧠 RLHF: Оцените ответы Sin (1–5)", style="bold yellow"))
    feedback = []

    # Простой датасет
    prompts = [
        "Привет",
        "Как дела?",
        "Расскажи анекдот",
        "Кто ты?",
        "Погода сегодня",
        "Сколько будет 2+2?",
        "Пока",
        "Что ты умеешь?"
    ]

    for prompt in random.sample(prompts, min(n, len(prompts))):
        input_text = f"Вопрос: {prompt}\nОтвет:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response[len(input_text):].strip() or "Не понял."

        console.print(f"[cyan]Вопрос:[/cyan] {prompt}")
        console.print(f"[magenta]Sin:[/magenta] {response}")
        rating = console.input("Оценка (1-5): ").strip()
        if rating in "12345":
            feedback.append({
                "input": input_text.strip(),
                "output": response,
                "score": int(rating)
            })

    # Сохраняем в JSONL
    mode = "a" if feedback_file.exists() else "w"
    with open(feedback_file, mode, encoding="utf-8") as f:
        for item in feedback:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"[green]Сохранено {len(feedback)} оценок.[/green]")

# ----------------------------------------
# Загрузка базовой модели (без ValueHead)
# ----------------------------------------
def load_model_base():
    global tokenizer, model
    model_path = MODEL_DIR / "finetuned_ppo"
    base_name = config["model_name"]

    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        logger.info("[blue]PPO-модель загружена.[/blue]")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLMWithValueHead.from_pretrained(base_name)
        logger.info("[yellow]Базовая модель загружена.[/yellow]")

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# RLHF: PPO обучение по оценкам
# ----------------------------------------
def train_from_feedback_ppo():
    feedback_file = DATA_DIR / "feedback.jsonl"
    if not os.path.exists(feedback_file):
        logger.warning("[yellow]Нет данных RLHF. Сначала соберите оценки.[/yellow]")
        return

    # Читаем оценки
    examples = []
    with open(feedback_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            examples.append({
                "prompt": item["input"].strip(),
                "response": item["output"].strip(),
                "reward": float(item["score"] - 3.0)  # нормализация: 1-5 → -2 до +2
            })

    if len(examples) < 2:
        logger.warning("[yellow]Нужно минимум 2 оценки для PPO.[/yellow]")
        return

    logger.info(f"[blue]Запуск PPO с {len(examples)} примерами...[/blue]")

    # Модель
    load_model_base()
    model_ppo = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model_ppo = get_peft_model(model_ppo, lora_config)

    # PPOTrainer
    ppo_config = PPOConfig(
        batch_size=2,
        mini_batch_size=2,
        learning_rate=1e-5,
        log_with=None,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model_ppo,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=None
    )

    # Данные
    prompts = [ex["prompt"] for ex in examples]
    rewards = [torch.tensor([ex["reward"]]) for ex in examples]

    # Токены
    prompt_tokens = [tokenizer.encode(p, return_tensors="pt").to(model.device) for p in prompts]

    # PPO шаг
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("PPO обучение...", total=len(prompts))
            for i in range(0, len(prompts), ppo_config.batch_size):
                batch_tokens = prompt_tokens[i:i + ppo_config.batch_size]
                batch_rewards = rewards[i:i + ppo_config.batch_size]

                # Генерация
                outputs = ppo_trainer.generate(
                    batch_tokens,
                    return_prompt=False,
                    gen_kwargs={
                        "min_length": -1,
                        "top_k": 0,
                        "top_p": 0.9,
                        "do_sample": True,
                        "max_new_tokens": 64,
                        "pad_token_id": tokenizer.eos_token_id,
                    },
                )
                # Убираем prompt
                texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

                # Реворды
                reward_tensors = [r.to(model.device) for r in batch_rewards]

                # Обучение
                stats = ppo_trainer.step(batch_tokens, outputs, reward_tensors)
                progress.update(task, advance=len(batch_tokens))

        # Сохранение
        model_ppo.pretrained_model.save_pretrained(MODEL_DIR / "finetuned_ppo")
        tokenizer.save_pretrained(MODEL_DIR / "finetuned_ppo")
        config["last_trained"] = datetime.now().isoformat()
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info("[bold green]PPO обучение завершено и сохранено.[/bold green]")

    except Exception as e:
        logger.error(f"[red]Ошибка PPO: {e}[/red]")

# ----------------------------------------
# Тест общения
# ----------------------------------------
def test_chat():
    load_model_base()
    memory.load(CACHE_DIR)
    console.print(Panel("💬 Sin — режим общения (PPO + RAG)", style="bold blue"))
    console.print("[yellow]Напишите 'exit' для выхода.[/yellow]")

    while True:
        user_input = console.input("[bold]Вы:[/bold] ")
        if user_input.lower() in ['exit', 'выход', 'quit']:
            break

        # RAG
        ctx = "\n".join([f"Ранее: {x}" for x in memory.search(user_input, k=2)])
        prompt = (ctx + "\n" if ctx else "") + f"Вопрос: {user_input}\nОтвет:"

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response[len(prompt):].strip() or "Я не понял."

        console.print(f"[magenta]Sin:[/magenta] {response}")

        memory.add(user_input)
        memory.add(response)

    memory.save(CACHE_DIR)

# ----------------------------------------
# Главное меню
# ----------------------------------------
def main():
    console.print(Panel("🤖 Sin — ИИ с RLHF и PPO", style="bold green"))
    while True:
        table = Table(title="Меню", show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim")
        table.add_column("Действие")
        table.add_row("1", "Собрать RLHF оценки (человек)")
        table.add_row("2", "PPO: дообучить по оценкам")
        table.add_row("3", "Тест общения (с памятью)")
        table.add_row("4", "Статус")
        table.add_row("5", "Выход")
        console.print(table)

        choice = console.input("[bold]Выберите: [/bold]")
        if choice == "1":
            collect_human_feedback(n=5)
        elif choice == "2":
            train_from_feedback_ppo()
        elif choice == "3":
            test_chat()
        elif choice == "4":
            status = Table(title="Статус Sin", show_header=True)
            status.add_column("Параметр")
            status.add_column("Значение")
            status.add_row("Последнее обучение", config.get("last_trained", "—"))
            fb_file = DATA_DIR / "feedback.jsonl"
            status.add_row("Оценок RLHF", str(sum(1 for _ in open(fb_file, 'r', encoding='utf-8')) if fb_file.exists() else 0))
            status.add_row("RAG записей", str(len(memory.sentences) if hasattr(memory, 'sentences') else 0))
            console.print(status)
        elif choice == "5":
            console.print("[bold red]До свидания, Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Неверная команда.[/red]")

if __name__ == "__main__":
    main()
