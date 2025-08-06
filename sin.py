# sin.py

import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from collections import deque
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
import logging

# --- ИМПОРТЫ ДЛЯ РАБОТЫ С ФАЙЛАМИ ---
try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("PyPDF2 не установлен. Поддержка PDF отключена.")
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("python-docx не установлен. Поддержка DOCX отключена.")

# -----------------------------------
# --- НОВЫЕ ИМПОРТЫ ДЛЯ ОБУЧЕНИЯ ПО URL ---
import requests
from bs4 import BeautifulSoup
import re
# ----------------------------------------

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
DATASETS_CACHE_DIR = PROJECT_DIR / "datasets_cache"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASETS_CACHE_DIR, exist_ok=True)

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

# --- Конфигурация для загрузки датасетов ---
HF_DATASETS = [
    {
        "name": "Russian QA Dataset",
        "path": "IlyaGusev/ru_qa",
        "config": None,
        "split": "train",
        "text_field": "text"
    },
    {
        "name": "Russian Conversational Dataset",
        "path": "IlyaGusev/ru_tweets",
        "config": None,
        "split": "train",
        "text_field": "text"
    }
]

GITHUB_DATASETS = [
    {
        "name": "Russian OpenSubtitles Dataset",
        "url": "https://github.com/akanyaani/rus-opensubtitles/releases/download/v1.0/opensubtitles_ru.txt.gz",
        "type": "gzip",
        "extract_path": "opensubtitles_ru.txt",
        "move_files": False
    },
    {
        "name": "Russian Wikipedia QA",
        "url": "https://github.com/avidale/ruqa/releases/download/v1.0/ruqa_wiki.jsonl.gz",
        "type": "gzip",
        "extract_path": "ruqa_wiki.jsonl",
        "move_files": False
    }
]

# ----------------------------------------
# Функции для загрузки датасетов
# ----------------------------------------
def download_and_extract_github_dataset(dataset_info: Dict) -> Optional[Path]:
    """Загружает и распаковывает датасет с GitHub"""
    import gzip
    import shutil
    from urllib.request import urlretrieve
    
    try:
        console.print(f"[blue]Загрузка датасета {dataset_info['name']}...[/blue]")
        archive_path = DATASETS_CACHE_DIR / Path(dataset_info['url']).name
        urlretrieve(dataset_info['url'], archive_path)
        
        if dataset_info['type'] == 'gzip':
            with gzip.open(archive_path, 'rb') as f_in:
                extracted_path = DATASETS_CACHE_DIR / dataset_info['extract_path']
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return extracted_path
        else:
            console.print(f"[red]Неподдерживаемый тип архива: {dataset_info['type']}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Ошибка при загрузке датасета: {e}[/red]")
        return None

def load_hf_dataset(dataset_info: Dict) -> Optional[List[Dict[str, str]]]:
    """Загружает датасет с Hugging Face"""
    try:
        from datasets import load_dataset
        console.print(f"[blue]Загрузка датасета {dataset_info['name']}...[/blue]")
        dataset = load_dataset(
            dataset_info['path'],
            dataset_info['config'] if dataset_info['config'] else None,
            split=dataset_info['split']
        )
        return [{"text": item[dataset_info['text_field']]} for item in dataset]
    except Exception as e:
        console.print(f"[red]Ошибка при загрузке датасета: {e}[/red]")
        return None

def load_text_from_txt(file_path: Path) -> str:
    """Загружает текст из .txt файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp1251') as f:
                return f.read()
        except Exception:
            pass
    except Exception:
        pass
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Ошибка при чтении {file_path}: {e}[/red]")
        return ""

def load_text_from_pdf(file_path: Path) -> str:
    """Загружает текст из .pdf файла."""
    if not HAS_PYPDF:
        console.print("[yellow]PyPDF2 не установлен. Невозможно прочитать PDF.[/yellow]")
        return ""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        return text
    except Exception as e:
        console.print(f"[red]Ошибка при чтении {file_path}: {e}[/red]")
        return ""

def load_text_from_docx(file_path: Path) -> str:
    """Загружает текст из .docx файла."""
    if not HAS_DOCX:
        console.print("[yellow]python-docx не установлен. Невозможно прочитать DOCX.[/yellow]")
        return ""
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        console.print(f"[red]Ошибка при чтении {file_path}: {e}[/red]")
        return ""

def load_qa_from_json(file_path: Path) -> List[Dict[str, str]]:
    """Загружает пары вопрос-ответ из .json файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        qa_pairs = []
        for item in data:
            if isinstance(item, dict):
                if "question" in item and "answer" in item:
                    qa_pairs.append({"text": f"{item['question']} Ответ: {item['answer']}"})
                elif "input" in item and "output" in item:
                    qa_pairs.append({"text": f"{item['input']} Ответ: {item['output']}"})
        return qa_pairs
    except Exception as e:
        console.print(f"[red]Ошибка при чтении {file_path}: {e}[/red]")
        return []

def load_dataset_from_files(data_dir: Path) -> List[Dict[str, str]]:
    """Загружает датасет из всех поддерживаемых файлов в директории."""
    all_qa_pairs = []
    if not data_dir.exists():
        console.print(f"[yellow]Директория {data_dir} не существует.[/yellow]")
        return all_qa_pairs
    supported_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf")) + \
                      list(data_dir.glob("*.docx")) + list(data_dir.glob("*.json"))
    if not supported_files:
        console.print(f"[yellow]В директории {data_dir} не найдено поддерживаемых файлов.[/yellow]")
        return all_qa_pairs
    for file_path in supported_files:
        console.print(f"[blue]Обработка файла: {file_path.name}[/blue]")
        if file_path.suffix.lower() == '.txt':
            text = load_text_from_txt(file_path)
            if text:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 500]
                for i in range(len(sentences) - 1):
                    if len(all_qa_pairs) >= 500:
                        break
                    question = sentences[i] + "?"
                    answer = sentences[i+1]
                    all_qa_pairs.append({"text": f"{question} Ответ: {answer}"})
        elif file_path.suffix.lower() == '.pdf':
            text = load_text_from_pdf(file_path)
            if text:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 500]
                for i in range(len(sentences) - 1):
                    if len(all_qa_pairs) >= 500:
                        break
                    question = sentences[i] + "?"
                    answer = sentences[i+1]
                    all_qa_pairs.append({"text": f"{question} Ответ: {answer}"})
        elif file_path.suffix.lower() == '.docx':
            text = load_text_from_docx(file_path)
            if text:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 500]
                for i in range(len(sentences) - 1):
                    if len(all_qa_pairs) >= 500:
                        break
                    question = sentences[i] + "?"
                    answer = sentences[i+1]
                    all_qa_pairs.append({"text": f"{question} Ответ: {answer}"})
        elif file_path.suffix.lower() == '.json':
            qa_pairs = load_qa_from_json(file_path)
            all_qa_pairs.extend(qa_pairs)
    console.print(f"[green]Загружено {len(all_qa_pairs)} пар вопрос-ответ из файлов.[/green]")
    return all_qa_pairs

# ----------------------------------------
# Функции для обучения по URL
# ----------------------------------------
def fetch_and_parse_url(url: str) -> Optional[str]:
    """Получает HTML по URL и извлекает основной текст."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "iframe", "meta"]):
            tag.decompose()
        main_content = soup.find('article') or soup.find('main') or \
                       soup.find('div', class_=re.compile(r'content|post|article|entry|text|story')) or \
                       soup.find('div', {'role': 'main'}) or \
                       soup.find('div', id=re.compile(r'content|main|article'))
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        console.print(f"[red]Ошибка при парсинге URL {url}: {e}[/red]")
        return None

def clean_text(text: str) -> str:
    """Базовая очистка текста."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    lines = [line.strip() for line in text.split('.') if len(line.strip()) > 15]
    cleaned_text = '. '.join(lines)
    cleaned_text = re.sub(r'[^\w\s.,!?;:()\-\nА-Яа-яёЁA-Za-z]', ' ', cleaned_text, flags=re.UNICODE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def create_qa_pairs_from_text(text: str, max_pairs: int = 100) -> List[Dict[str, str]]:
    """Создает пары вопрос-ответ из текста."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if 15 < len(s.strip()) < 500] 
    qa_pairs = []
    for i in range(len(sentences) - 1):
        if len(qa_pairs) >= max_pairs:
            break
        statement = sentences[i]
        next_statement = sentences[i+1]
        if statement.endswith('.'):
            question = statement[:-1] + '?' 
        else:
            question = statement + '?'
        answer = next_statement
        formatted_text = f"{question} Ответ: {answer}"
        qa_pairs.append({"text": formatted_text})
    console.print(f"[green]Создано {len(qa_pairs)} пар вопрос-ответ.[/green]")
    return qa_pairs

def prepare_dataset_from_url(url: str, max_pairs: int = 100) -> Optional[List[Dict[str, str]]]:
    """Основная функция для подготовки датасета из URL."""
    console.print(f"[blue]Начинаем парсинг URL: {url}[/blue]")
    raw_text = fetch_and_parse_url(url)
    if not raw_text:
        return None
    console.print("[blue]Очистка текста...[/blue]")
    clean_text_content = clean_text(raw_text)
    if not clean_text_content or len(clean_text_content) < 100:
        console.print("[red]Очищенный текст слишком мал или пуст.[/red]")
        return None
    console.print("[blue]Создание пар вопрос-ответ...[/blue]")
    qa_pairs = create_qa_pairs_from_text(clean_text_content, max_pairs)
    if not qa_pairs:
        console.print("[red]Не удалось создать пары вопрос-ответ.[/red]")
        return None
    console.print(f"[green]Датасет из URL успешно создан. Размер: {len(qa_pairs)} примеров.[/green]")
    return qa_pairs

# ----------------------------------------
# Загрузка модели с оптимизацией и автозагрузкой
# ----------------------------------------
def load_optimized_model(auto_load: bool = True, model_path_or_name: str = "cointegrated/rut5-base"):
    """
    Загружает модель. Если auto_load=True и модель не найдена локально,
    пытается загрузить её из Hugging Face.
    """
    local = (MODEL_DIR / "config.json").exists()
    try:
        if not local and not auto_load:
            console.print("[yellow]Локальная модель не найдена и автозагрузка отключена.[/yellow]")
            return None, None
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR if local else model_path_or_name,
            local_files_only=local,
            use_fast=False,
            legacy=False # Используем новое поведение токенизатора
        )
        console.print(f"[green]Токенизатор загружен из: {MODEL_DIR if local else model_path_or_name}[/green]")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR if local else model_path_or_name,
            local_files_only=local,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        console.print(f"[green]Модель загружена из: {MODEL_DIR if local else model_path_or_name}[/green]")
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
        positive_words = ["хорош", "прекрасн", "рад", "счастлив", "люблю", "отлично", "замечательно", "прекрасно"]
        negative_words = ["плох", "ужасн", "грустн", "злюсь", "ненавижу", "печаль", "отстой", "фу"]
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
        sentence_model_name = "all-MiniLM-L6-v2"
        sentence_model_path = Path(sentence_model_name)
        if sentence_model_path.exists() and sentence_model_path.is_dir():
            sentence_model_source = str(sentence_model_path)
        else:
            sentence_model_source = sentence_model_name
        self.sentence_model = SentenceTransformer(sentence_model_source, device=device)
        self.memories = []
        if self.memory_file.exists():
            self.load_memory()

    def add_memory(self, text: str, emotion_state: Dict):
        try:
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
        except Exception as e:
            console.print(f"[red]Ошибка при добавлении в память: {e}[/red]")

    def find_related_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.memories:
            return []
        try:
            query_embedding = self.sentence_model.encode(query)
            memory_embeddings = np.array([np.array(m["embedding"]) for m in self.memories])
            similarities = cosine_similarity([query_embedding], memory_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.memories[i] for i in top_indices]
        except Exception as e:
            console.print(f"[red]Ошибка при поиске в памяти: {e}[/red]")
            return []

    def save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump({"memories": self.memories}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[red]Ошибка при сохранении памяти: {e}[/red]")

    def load_memory(self):
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.memories = json.load(f).get("memories", [])
        except Exception as e:
            console.print(f"[red]Ошибка при загрузке памяти: {e}[/red]")
            self.memories = []

# --- PyTorch Dataset для обучения ---
class QADataset(Dataset):
    def __init__(self, qa_pairs: List[Dict[str, str]], tokenizer, max_length: int = 256):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        text = item['text']
        if "Ответ:" in text:
            parts = text.split("Ответ:", 1)
            input_text = parts[0].strip()
            target_text = parts[1].strip()
        else:
            input_text = text
            target_text = "Хорошо, я понял."
        
        # Исправление для устранения предупреждения об устаревшем методе
        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # Исправление для устранения предупреждения об устаревшем методе
        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]
        
        model_inputs = {key: val.squeeze(0) for key, val in model_inputs.items()}
        labels = labels.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        
        # Исправление для устранения предупреждения о медленном создании тензора
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].clone().detach()
            else:
                model_inputs[key] = torch.tensor(model_inputs[key], dtype=torch.long)
        
        return model_inputs

# --- Основной класс бота ---
class EmotionalChatBot:
    def __init__(self, auto_load_model: bool = True):
        self.tokenizer, self.model = load_optimized_model(auto_load=auto_load_model)
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
        """Сохраняет конфигурацию бота и модель."""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            if self.model is not None and self.tokenizer is not None:
                self.model.save_pretrained(MODEL_DIR)
                self.tokenizer.save_pretrained(MODEL_DIR)
                console.print("[green]Модель успешно сохранена.[/green]")
            else:
                console.print("[yellow]Нет модели для сохранения.[/yellow]")
        except Exception as e:
            console.print(f"[red]Ошибка при сохранении состояния: {e}[/red]")

# ----------------------------------------
# Обучение с PyTorch (веса, слои, обратное распространение)
# ----------------------------------------
def train_pytorch(bot: EmotionalChatBot, train_dataset: Dataset, epochs: int = 3, batch_size: int = 2, learning_rate: float = 5e-5, save_steps: int = 10):
    """Обучение модели с использованием PyTorch."""
    if bot.model is None or bot.tokenizer is None:
        logger.warning("[yellow]Нет модели или токенизатора для обучения.[/yellow]")
        bot.tokenizer, bot.model = load_optimized_model(auto_load=True)
        if bot.model is None or bot.tokenizer is None:
            console.print("[red]Не удалось загрузить модель для обучения.[/red]")
            return

    console.print("[blue]Запуск PyTorch SFT обучения...[/blue]")
    
    try:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        if not isinstance(bot.model, PeftModel):
            model = get_peft_model(bot.model, peft_config)
        else:
            model = bot.model
        model.print_trainable_parameters()
    except Exception as e:
        console.print(f"[red]Ошибка при настройке LoRA: {e}[/red]")
        return

    try:
        data_collator = DataCollatorForSeq2Seq(bot.tokenizer, model=model, padding=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    except Exception as e:
        console.print(f"[red]Ошибка при подготовке данных: {e}[/red]")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(device)
    model.train()

    console.print(f"[blue]Обучение на {len(train_dataset)} примерах, {epochs} эпох, batch_size={batch_size}[/blue]")
    progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console)
    
    with progress:
        task = progress.add_task("[green]Обучение...", total=total_steps)
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for step, batch in enumerate(train_dataloader):
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    num_batches += 1
                    progress.update(task, description=f"[green]Эпоха {epoch+1}/{epochs}[/green] - Loss: {loss.item():.4f}")
                    progress.advance(task)
                    if (step + 1) % save_steps == 0:
                        console.print(f"[yellow]Промежуточное сохранение на шаге {step+1}...[/yellow]")
                        adapter_dir = MODEL_DIR / f"adapter_checkpoint_step_{step+1}"
                        os.makedirs(adapter_dir, exist_ok=True)
                        model.save_pretrained(adapter_dir)
                        bot.tokenizer.save_pretrained(adapter_dir)
                        console.print(f"[green]Адаптеры сохранены в {adapter_dir}[/green]")
                except Exception as e:
                    console.print(f"[red]Ошибка на шаге {step}: {e}[/red]")
                    continue
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            console.print(f"[green]Эпоха {epoch+1} завершена. Средний Loss: {avg_epoch_loss:.4f}[/green]")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    try:
        console.print("[blue]Сохранение обученной модели...[/blue]")
        model.save_pretrained(MODEL_DIR)
        bot.tokenizer.save_pretrained(MODEL_DIR)
        bot.config["last_trained"] = datetime.now().isoformat()
        bot.save_state()
        console.print("[green]PyTorch SFT обучение завершено и модель сохранена.[/green]")
    except Exception as e:
        console.print(f"[red]Ошибка при сохранении модели: {e}[/red]")

# ----------------------------------------
# SFT: Обучение на парах вопрос-ответ (с улучшениями)
# ----------------------------------------
def train_sft(bot: EmotionalChatBot, custom_dataset: Optional[List[Dict[str, str]]] = None, dataset_source: str = None):
    """Обучение с использованием SFTTrainer."""
    if bot.model is None or bot.tokenizer is None:
        logger.warning("[yellow]Нет модели или токенизатора для обучения.[/yellow]")
        bot.tokenizer, bot.model = load_optimized_model(auto_load=True)
        if bot.model is None or bot.tokenizer is None:
            console.print("[red]Не удалось загрузить модель для обучения.[/red]")
            return

    data = []
    if custom_dataset is not None:
        data = custom_dataset
        console.print("[blue]Используется пользовательский датасет для обучения.[/blue]")
    elif dataset_source == "hf":
        console.print("[blue]Загрузка датасета с Hugging Face...[/blue]")
        for dataset_info in HF_DATASETS:
            hf_data = load_hf_dataset(dataset_info)
            if hf_data:
                data.extend(hf_data)
                console.print(f"[green]Загружено {len(hf_data)} примеров из {dataset_info['name']}[/green]")
                break
    elif dataset_source == "github":
        console.print("[blue]Загрузка датасета с GitHub...[/blue]")
        for dataset_info in GITHUB_DATASETS:
            dataset_path = download_and_extract_github_dataset(dataset_info)
            if dataset_path:
                if dataset_path.suffix == '.jsonl':
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            if 'question' in item and 'answer' in item:
                                data.append({"text": f"{item['question']} Ответ: {item['answer']}"})
                elif dataset_path.suffix == '.txt':
                    text = load_text_from_txt(dataset_path)
                    if text:
                        qa_pairs = create_qa_pairs_from_text(text, max_pairs=500)
                        data.extend(qa_pairs)
                console.print(f"[green]Загружено {len(data)} примеров из {dataset_info['name']}[/green]")
                break
    else:
        console.print("[blue]Используется стандартный датасет для обучения.[/blue]")
        data = [
            {"text": "Привет Ответ: Здравствуй! Как дела?"},
            {"text": "Как дела? Ответ: У меня всё хорошо, спасибо!"},
            {"text": "Расскажи анекдот Ответ: Почему программисты не ходят в лес? Боятся рекурсии!"},
            {"text": "Что ты умеешь? Ответ: Я могу поддержать беседу, рассказать анекдот и помочь с различными вопросами."},
            {"text": "Как тебя зовут? Ответ: Меня зовут Sin. Приятно познакомиться!"},
            {"text": "Пока Ответ: До скорой встречи!"},
            {"text": "Что такое ИИ? Ответ: Искусственный интеллект - это область компьютерных наук, которая создает интеллектуальные машины."},
            {"text": "Расскажи о погоде Ответ: Я не могу получить информацию о погоде в реальном времени, но могу поговорить о климате."},
        ]
    
    if not data:
        console.print("[red]Не удалось загрузить данные для обучения.[/red]")
        return

    try:
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(data)
    except ImportError:
        console.print("[red]Библиотека datasets не установлена. Установите её: pip install datasets[/red]")
        return

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    from copy import deepcopy
    try:
        model_for_training = deepcopy(bot.model)
        model_for_training = get_peft_model(model_for_training, peft_config)
        model_for_training.print_trainable_parameters()
    except Exception as e:
        console.print(f"[red]Ошибка при настройке LoRA для SFT: {e}[/red]")
        return

    training_args = TrainingArguments(
        output_dir=str(LOGS_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        logging_first_step=True,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
    )

    try:
        trainer = SFTTrainer(
            model=model_for_training,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=256,
            packing=False,
            tokenizer=bot.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=bot.tokenizer, 
                model=model_for_training, 
                padding=True,
                return_tensors="pt"  # Исправление для предупреждения о медленных тензорах
            )
        )
    except Exception as e:
        console.print(f"[red]Ошибка при создании SFTTrainer: {e}[/red]")
        return
        
    console.print("[blue]Запуск процесса обучения SFT...[/blue]")
    try:
        with Progress(SpinnerColumn(), TextColumn("SFT обучение..."), BarColumn(), console=console) as progress:
            progress.add_task("", total=None)
            trainer.train()
    except Exception as e:
        console.print(f"[red]Ошибка во время обучения SFT: {e}[/red]")
        return
        
    console.print("[green]Сохранение обученной модели...[/green]")
    try:
        model_for_training.save_pretrained(MODEL_DIR)
        bot.config["last_trained"] = datetime.now().isoformat()
        bot.model = model_for_training.merge_and_unload()
        bot.save_state()
        logger.info("[bold green]SFT обучение завершено и модель обновлена.[/bold green]")
    except Exception as e:
        console.print(f"[red]Ошибка при сохранении модели после SFT: {e}[/red]")

# ----------------------------------------
# RLHF: Сбор оценок
# ----------------------------------------
def collect_rlhf_feedback(bot: EmotionalChatBot):
    feedback_file = DATA_DIR / "feedback.jsonl"
    console.print(Panel("🧠 Оцените ответы Sin (1–5)", style="bold yellow"))
    feedback = []
    prompts = [
        "Привет", "Как дела?", "Расскажи анекдот", "Кто ты?", "Погода", 
        "2+2", "Пока", "Что нового?", "Спасибо", "Расскажи о себе",
        "Как настроение?", "Что ты думаешь о людях?", "Расскажи историю"
    ]
    num_samples = min(5, len(prompts))
    for q in random.sample(prompts, num_samples): 
        response = bot.generate_response(q)
        console.print(f"[cyan]Вопрос:[/cyan] {q}")
        console.print(f"[magenta]Sin:[/magenta] {response}")
        while True:
            rating = console.input("Оценка (1-5): ").strip()
            if rating.isdigit() and 1 <= int(rating) <= 5:
                feedback.append({"input": q, "output": response, "score": int(rating)})
                break
            else:
                console.print("[red]Пожалуйста, введите число от 1 до 5.[/red]")
    if feedback:
        try:
            with open(feedback_file, "a", encoding="utf-8") as f:
                for item in feedback:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"[green]Сохранено {len(feedback)} оценок.[/green]")
        except Exception as e:
            logger.error(f"[red]Ошибка при сохранении оценок: {e}[/red]")
    else:
        logger.info("[yellow]Оценки не были собраны.[/yellow]")

# ----------------------------------------
# Главное меню
# ----------------------------------------
def main():
    console.print(Panel.fit("🤖 [bold green]Sin — ваш эмоциональный ассистент[/bold green]"))
    console.print("Напишите 'выход' для завершения диалога\n")
    bot = EmotionalChatBot(auto_load_model=True)
    while True:
        table = Table(title="Меню", show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim")
        table.add_column("Действие")
        table.add_row("1", "Поговорить с Sin")
        table.add_row("2", "Собрать RLHF оценки")
        table.add_row("3", "SFT: дообучить модель (стандартные данные)")
        table.add_row("4", "SFT: дообучить модель по URL")
        table.add_row("5", "SFT: дообучить модель из файлов")
        table.add_row("6", "PyTorch: дообучить модель из файлов")
        table.add_row("7", "SFT: дообучить модель с Hugging Face")
        table.add_row("8", "SFT: дообучить модель с GitHub")
        table.add_row("9", "Сохранить состояние")
        table.add_row("10", "Выход")
        console.print(table)
        choice = console.input("[bold]Выберите: [/bold]").strip()
        if choice == "1":
            if bot.model is None:
                 console.print("[red]Модель не загружена. Пожалуйста, сначала обучите её или включите автозагрузку.[/red]")
                 continue
            while True:
                user_input = console.input("[bold blue]Вы:[/bold blue] ").strip()
                if user_input.lower() in ("выход", "exit", "quit"):
                    break
                if not user_input:
                    console.print("[yellow]Пожалуйста, введите сообщение.[/yellow]")
                    continue
                response = bot.generate_response(user_input)
                console.print(f"[bold magenta]Sin:[/bold magenta] {response}")
                emotion_state = bot.emotion_engine.get_state()
                table = Table(title="Состояние", show_header=False)
                table.add_row("Эмоция", f"{emotion_state['emotion_icon']} {emotion_state['current_emotion']}")
                table.add_row("Интенсивность", f"{emotion_state['intensity']:.2f}")
                mood_desc = "😊 Хорошее" if emotion_state['long_term_mood'] > 0.3 else "😐 Нейтральное" if -0.3 <= emotion_state['long_term_mood'] <= 0.3 else "😢 Плохое"
                table.add_row("Настроение", mood_desc)
                console.print(table)
        elif choice == "2":
            collect_rlhf_feedback(bot)
        elif choice == "3":
            train_sft(bot)
        elif choice == "4":
            url = console.input("Введите URL для обучения: ").strip()
            if url:
                console.print("[blue]Подготовка датасета из URL...[/blue]")
                dataset_from_url = prepare_dataset_from_url(url, max_pairs=50) 
                if dataset_from_url:
                    console.print("[blue]Запуск обучения на данных из URL...[/blue]")
                    train_sft(bot, custom_dataset=dataset_from_url)
                else:
                    console.print("[red]Не удалось подготовить датасет из указанного URL.[/red]")
            else:
                console.print("[red]URL не был введен.[/red]")
        elif choice == "5":
            console.print("[blue]Загрузка датасета из файлов...[/blue]")
            qa_pairs = load_dataset_from_files(DATA_DIR)
            if qa_pairs:
                console.print("[blue]Запуск SFT обучения на данных из файлов...[/blue]")
                train_sft(bot, custom_dataset=qa_pairs)
            else:
                console.print("[red]Не удалось загрузить датасет из файлов.[/red]")
        elif choice == "6":
            console.print("[blue]Загрузка датасета из файлов для PyTorch обучения...[/blue]")
            qa_pairs = load_dataset_from_files(DATA_DIR)
            if qa_pairs and bot.tokenizer:
                console.print("[blue]Создание PyTorch Dataset...[/blue]")
                try:
                    dataset = QADataset(qa_pairs, bot.tokenizer)
                    console.print("[blue]Запуск PyTorch обучения...[/blue]")
                    train_pytorch(bot, dataset, epochs=2, batch_size=1, learning_rate=3e-5, save_steps=20)
                except Exception as e:
                    console.print(f"[red]Ошибка при создании датасета или запуске обучения: {e}[/red]")
            else:
                console.print("[red]Не удалось загрузить датасет или токенизатор.[/red]")
        elif choice == "7":
            console.print("[blue]Запуск обучения на датасете с Hugging Face...[/blue]")
            train_sft(bot, dataset_source="hf")
        elif choice == "8":
            console.print("[blue]Запуск обучения на датасете с GitHub...[/blue]")
            train_sft(bot, dataset_source="github")
        elif choice == "9":
             bot.save_state()
             console.print("[green]Состояние сохранено.[/green]")
        elif choice == "10":
            bot.save_state()
            console.print("[bold red]До свидания, Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Ошибка. Неверный выбор.[/red]")

if __name__ == "__main__":
    main()
