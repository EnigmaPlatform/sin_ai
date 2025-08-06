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
from peft import get_peft_model, LoraConfig, TaskType
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
# Функции для загрузки датасетов из файлов
# ----------------------------------------
def load_text_from_txt(file_path: Path) -> str:
    """Загружает текст из .txt файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Попробуем другую кодировку
        try:
            with open(file_path, 'r', encoding='cp1251') as f:
                return f.read()
        except Exception:
            pass
    except Exception:
        pass
    # Последняя попытка без указания кодировки
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
    """Загружает пары вопрос-ответ из .json файла.
    Ожидается формат: [{"question": "...", "answer": "..."}, ...] или [{"input": "...", "output": "..."}, ...]
    """
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
                # Можно добавить другие форматы
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
                # Создаем QA пары из текста
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 500]
                for i in range(len(sentences) - 1):
                    if len(all_qa_pairs) >= 500: # Ограничение для производительности
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
        
        # Более агрессивная очистка
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "iframe", "meta"]):
            tag.decompose()
            
        # Попытка найти основной контент
        main_content = soup.find('article') or soup.find('main') or \
                       soup.find('div', class_=re.compile(r'content|post|article|entry|text|story')) or \
                       soup.find('div', {'role': 'main'}) or \
                       soup.find('div', id=re.compile(r'content|main|article'))
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Если не найдено, берем текст основного тела
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
    # Удаление лишних пробелов и переносов строк
    text = re.sub(r'\s+', ' ', text)
    # Удаление очень коротких строк
    lines = [line.strip() for line in text.split('.') if len(line.strip()) > 15]
    cleaned_text = '. '.join(lines)
    # Удаление остаточных непечатаемых символов (оставляем кириллицу, латиницу и базовую пунктуацию)
    cleaned_text = re.sub(r'[^\w\s.,!?;:()\-\nА-Яа-яёЁA-Za-z]', ' ', cleaned_text, flags=re.UNICODE)
    # Еще одна очистка от лишних пробелов после удаления символов
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def create_qa_pairs_from_text(text: str, max_pairs: int = 100) -> List[Dict[str, str]]:
    """Создает пары вопрос-ответ из текста."""
    # Разбиваем на предложения
    sentences = re.split(r'[.!?]+', text)
    # Фильтруем слишком короткие и длинные предложения
    sentences = [s.strip() for s in sentences if 15 < len(s.strip()) < 500] 
    
    qa_pairs = []
    for i in range(len(sentences) - 1):
        if len(qa_pairs) >= max_pairs:
            break
        # Создаем искусственную пару "Вопрос -> Ответ"
        statement = sentences[i]
        next_statement = sentences[i+1]
        
        # Простая эвристика для создания вопроса: заменяем точку на вопросительный знак
        if statement.endswith('.'):
            question = statement[:-1] + '?' 
        else:
            question = statement + '?'
            
        answer = next_statement
        # Формат, как в вашем исходном примере
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
    if not clean_text_content or len(clean_text_content) < 100: # Проверка на минимальный объем
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
def load_optimized_model(auto_load: bool = True):
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
        # Проверка наличия модели SentenceTransformer локально
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
            similarities = cosine_similarity([query_embedding], [np.array(m["embedding"]) for m in self.memories])[0]
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
        
        # Разделяем на вход и цель
        if "Ответ:" in text:
            parts = text.split("Ответ:", 1)
            input_text = parts[0].strip()
            target_text = parts[1].strip()
        else:
            # Если формат не соответствует, используем весь текст как вход
            input_text = text
            target_text = "Хорошо, я понял."

        # Токенизация
        model_inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        # Убираем batch dimension
        model_inputs = {key: val.squeeze(0) for key, val in model_inputs.items()}
        labels = {key: val.squeeze(0) for key, val in labels.items()}
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# --- Основной класс бота ---
class EmotionalChatBot:
    def __init__(self, auto_load_model: bool = True):
        # ✅ Автозагрузка модели при инициализации
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
            # ✅ Автосохранение модели
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
def train_pytorch(bot: EmotionalChatBot, train_dataset: Dataset, epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5):
    """Обучение модели с использованием PyTorch."""
    if bot.model is None:
        logger.warning("[yellow]Нет модели для обучения.[/yellow]")
        # Попробуем перезагрузить
        bot.tokenizer, bot.model = load_optimized_model(auto_load=True)
        if bot.model is None:
            console.print("[red]Не удалось загрузить модель для обучения.[/red]")
            return

    # LoRA адаптация
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Создаем копию модели для обучения
    from copy import deepcopy
    try:
        model_for_training = deepcopy(bot.model)
        model_for_training = get_peft_model(model_for_training, peft_config)
        model_for_training.print_trainable_parameters()
    except Exception as e:
        console.print(f"[red]Ошибка при настройке LoRA: {e}[/red]")
        return
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model_for_training.parameters(), lr=learning_rate)
    
    # DataLoader
    try:
        data_collator = DataCollatorForSeq2Seq(bot.tokenizer, model=model_for_training, padding=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    except Exception as e:
        console.print(f"[red]Ошибка при создании DataLoader: {e}[/red]")
        return
    
    model_for_training.to(device)
    model_for_training.train()
    
    console.print(f"[blue]Запуск PyTorch обучения на {epochs} эпохах...[/blue]")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        with Progress(SpinnerColumn(), TextColumn(f"Эпоха {epoch+1}/{epochs}"), BarColumn(), console=console) as progress:
            task = progress.add_task("", total=len(train_dataloader))
            
            for batch in train_dataloader:
                try:
                    # Перемещаем батч на устройство
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Прямой проход
                    outputs = model_for_training(**batch)
                    loss = outputs.loss
                    
                    # Обратный проход
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Ошибка в батче: {e}[/red]")
                    progress.update(task, advance=1)
                    continue
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        console.print(f"[green]Эпоха {epoch+1} завершена. Средняя потеря: {avg_loss:.4f}[/green]")
    
    console.print("[green]Сохранение обученной модели...[/green]")
    try:
        model_for_training.save_pretrained(MODEL_DIR)
        bot.config["last_trained"] = datetime.now().isoformat()
        # Обновляем модель бота
        bot.model = model_for_training.merge_and_unload() # Объединяем LoRA веса с основной моделью
        bot.save_state() # Сохраняем обновленную модель
        logger.info("[bold green]PyTorch обучение завершено и модель обновлена.[/bold green]")
    except Exception as e:
        console.print(f"[red]Ошибка при сохранении модели после обучения: {e}[/red]")

# ----------------------------------------
# SFT: Обучение на парах вопрос-ответ (с улучшениями)
# ----------------------------------------
def train_sft(bot: EmotionalChatBot, custom_dataset: Optional[List[Dict[str, str]]] = None):
    """Обучение с использованием SFTTrainer."""
    if bot.model is None:
        logger.warning("[yellow]Нет модели для обучения.[/yellow]")
        bot.tokenizer, bot.model = load_optimized_model(auto_load=True)
        if bot.model is None:
            console.print("[red]Не удалось загрузить модель для обучения.[/red]")
            return
    
    # Подготовка данных
    if custom_dataset is not None:
        data = custom_dataset
        console.print("[blue]Используется пользовательский датасет для обучения.[/blue]")
    else:
        # Пример данных по умолчанию, если датасет не передан
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

    try:
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(data)
    except ImportError:
        console.print("[red]Библиотека datasets не установлена. Установите её: pip install datasets[/red]")
        return

    # LoRA
    # ✅ Исправлен вызов get_peft_model с правильной конфигурацией TaskType
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"], # Целевые модули для rut5-base
        task_type=TaskType.SEQ_2_SEQ_LM # ✅ Указание типа задачи
    )
    # Создаем копию модели для обучения, чтобы не изменять оригинальную
    from copy import deepcopy
    try:
        model_for_training = deepcopy(bot.model)
        model_for_training = get_peft_model(model_for_training, peft_config)
        model_for_training.print_trainable_parameters() # Печатаем информацию о параметрах
    except Exception as e:
        console.print(f"[red]Ошибка при настройке LoRA для SFT: {e}[/red]")
        return

    # Тренировка
    training_args = TrainingArguments(
        output_dir=str(LOGS_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, # Эффективный batch size = 16
        num_train_epochs=3, # Увеличено для лучшего обучения
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=None,
        dataloader_pin_memory=False, # Может помочь с ошибками на CPU
        remove_unused_columns=True,
        logging_first_step=True,
        # save_strategy="steps",
        # evaluation_strategy="no", # или "steps" если есть eval_dataset
        # Добавлены для стабильности
        load_best_model_at_end=False, # Нет валидации, поэтому отключено
        dataloader_num_workers=0, # Может помочь с ошибками в Windows
    )

    # ✅ Использование DataCollatorForSeq2Seq для корректной обработки данных
    try:
        trainer = SFTTrainer(
            model=model_for_training,
            args=training_args,
            train_dataset=dataset,
            tokenizer=bot.tokenizer,
            dataset_text_field="text",
            max_seq_length=256,
            packing=False, # Отключаем упаковку для простоты
            data_collator=DataCollatorForSeq2Seq(tokenizer=bot.tokenizer, model=model_for_training, padding=True)
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
        # После обучения сохраняем адаптеры LoRA
        model_for_training.save_pretrained(MODEL_DIR)
        # Токенизатор не нужно сохранять снова, если он не изменился
        # bot.tokenizer.save_pretrained(MODEL_DIR) 
        bot.config["last_trained"] = datetime.now().isoformat()
        # ✅ Обновляем модель бота на обученную версию
        bot.model = model_for_training.merge_and_unload() # Объединяем LoRA веса с основной моделью
        bot.save_state() # Сохраняем обновленную модель
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
    # Увеличено количество примеров для оценки
    num_samples = min(5, len(prompts))
    for q in random.sample(prompts, num_samples): 
        response = bot.generate_response(q)
        console.print(f"[cyan]Вопрос:[/cyan] {q}")
        console.print(f"[magenta]Sin:[/magenta] {response}")
        while True: # Цикл для проверки ввода
            rating = console.input("Оценка (1-5): ").strip()
            if rating.isdigit() and 1 <= int(rating) <= 5:
                feedback.append({"input": q, "output": response, "score": int(rating)})
                break
            else:
                console.print("[red]Пожалуйста, введите число от 1 до 5.[/red]")
                
    if feedback: # Сохраняем только если есть оценки
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
    # ✅ Автозагрузка модели при запуске
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
        table.add_row("7", "Сохранить состояние")
        table.add_row("8", "Выход")
        console.print(table)
        choice = console.input("[bold]Выберите: [/bold]").strip() # Убран пробел в конце
        if choice == "1":
            if bot.model is None:
                 console.print("[red]Модель не загружена. Пожалуйста, сначала обучите её или включите автозагрузку.[/red]")
                 continue
            while True:
                user_input = console.input("[bold blue]Вы:[/bold blue] ").strip()
                if user_input.lower() in ("выход", "exit", "quit"):
                    break
                if not user_input: # Проверка на пустой ввод
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
            train_sft(bot) # Обучение на стандартных данных
        elif choice == "4":
            url = console.input("Введите URL для обучения: ").strip()
            if url:
                console.print("[blue]Подготовка датасета из URL...[/blue]")
                # Уменьшено количество пар для быстрого тестирования
                dataset_from_url = prepare_dataset_from_url(url, max_pairs=50) 
                if dataset_from_url:
                    console.print("[blue]Запуск обучения на данных из URL...[/blue]")
                    train_sft(bot, custom_dataset=dataset_from_url) # Передаем подготовленный датасет
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
                    train_pytorch(bot, dataset)
                except Exception as e:
                    console.print(f"[red]Ошибка при создании датасета или запуске обучения: {e}[/red]")
            else:
                console.print("[red]Не удалось загрузить датасет или токенизатор.[/red]")
        elif choice == "7":
             bot.save_state()
             console.print("[green]Состояние сохранено.[/green]")
        elif choice == "8":
            # ✅ Автосохранение при выходе
            bot.save_state()
            console.print("[bold red]До свидания, Sin спит...[/bold red]")
            break
        else:
            console.print("[red]Ошибка. Неверный выбор.[/red]")

if __name__ == "__main__":
    main()
