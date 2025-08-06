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
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
import logging
import traceback
import sys
import re
import gc
from typing import Any

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

# --- НАСТРОЙКИ ПУТЕЙ ---
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BASE_DIR / "Sin"
MODEL_DIR = PROJECT_DIR / "model"
DATA_DIR = PROJECT_DIR / "data"
MEMORY_DIR = PROJECT_DIR / "memory"
CONFIG_FILE = PROJECT_DIR / "config.json"
LOGS_DIR = PROJECT_DIR / "logs"
DATASETS_CACHE_DIR = PROJECT_DIR / "datasets_cache"
RLHF_QUESTIONS_FILE = DATA_DIR / "rlhf_questions.txt"

# Создание директорий
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASETS_CACHE_DIR, exist_ok=True)

# --- НАСТРОЙКА ЛОГГИРОВАНИЯ ---
rich_console = Console(width=120)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=rich_console, show_path=False)]
)
logger = logging.getLogger("Sin")
console = Console()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- КОНФИГУРАЦИЯ ---
EMOTION_ENGINE_CONFIG = {
    "base_emotions": {
        "happy": {"icon": "😊", "triggers": ["рад", "счастлив", "люблю", "хорош", "прекрасн"]},
        "sad": {"icon": "😢", "triggers": ["грустн", "печаль", "плак", "плох", "ужасн"]},
        "angry": {"icon": "😠", "triggers": ["злюсь", "бесит", "ненавижу", "злой", "разозлился"]},
        "fear": {"icon": "😨", "triggers": ["боюсь", "страх", "пугает", "испуг", "опасно"]},
        "surprise": {"icon": "😲", "triggers": ["невероятно", "удивитель", "ого", "вау", "неожиданно"]},
        "disgust": {"icon": "🤢", "triggers": ["отврат", "мерзк", "противно", "фу", "гадость"]},
        "neutral": {"icon": "😐", "triggers": []}
    },
    "decay_rate": 0.95,
    "intensity_threshold": 0.3,
    "max_memory": 1000
}

# --- КЛАСС ЭМОЦИОНАЛЬНОГО СОСТОЯНИЯ ---
class EmotionalState:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.5
        self.emotion_history = deque(maxlen=50)
        self.long_term_mood = 0.5
        self.triggers_activated = set()
        self.sentiment_analyzer = None
        self._initialize_sentiment_analyzer()

    def _initialize_sentiment_analyzer(self):
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                device=device if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Не удалось загрузить анализатор тональности: {e}")
            self.sentiment_analyzer = None

    def update(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            logger.warning("Пустой текст для эмоционального анализа")
            return

        try:
            text_lower = text.lower()
            
            # Обновление эмоций на основе триггеров
            emotion_scores = {e: 0 for e in EMOTION_ENGINE_CONFIG["base_emotions"]}
            
            for emotion, data in EMOTION_ENGINE_CONFIG["base_emotions"].items():
                for trigger in data["triggers"]:
                    if trigger in text_lower:
                        emotion_scores[emotion] += 1
            
            # Определение доминирующей эмоции
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            if emotion_scores[max_emotion] > 0:
                self.current_emotion = max_emotion
                self.emotion_intensity = min(1.0, self.emotion_intensity + 0.3 * emotion_scores[max_emotion])
                self.triggers_activated.add(max_emotion)
            
            # Постепенное затухание эмоций
            self.emotion_intensity = max(0, self.emotion_intensity * EMOTION_ENGINE_CONFIG["decay_rate"])
            if self.emotion_intensity < EMOTION_ENGINE_CONFIG["intensity_threshold"]:
                self.current_emotion = "neutral"
            
            # Анализ настроения
            sentiment = self.analyze_sentiment(text)
            self.long_term_mood = max(-1.0, min(1.0, 
                0.9 * self.long_term_mood + 0.1 * sentiment))
            
            # Сохранение истории
            self.emotion_history.append({
                "timestamp": datetime.now().isoformat(),
                "emotion": self.current_emotion,
                "intensity": self.emotion_intensity,
                "text": text[:200]  # Сохраняем только начало текста
            })
            
        except Exception as e:
            logger.error(f"Ошибка в EmotionalState.update: {e}\n{traceback.format_exc()}")
            self.current_emotion = "neutral"
            self.emotion_intensity = 0.5

    def analyze_sentiment(self, text: str) -> float:
        if not text or not isinstance(text, str):
            return 0.0
            
        try:
            if self.sentiment_analyzer is None:
                return self._fallback_sentiment_analysis(text)
            
            result = self.sentiment_analyzer(text[:512])[0]  # Ограничение длины
            if result['label'] == 'POSITIVE':
                return result['score']
            elif result['label'] == 'NEGATIVE':
                return -result['score']
            return 0.0
            
        except Exception as e:
            logger.error(f"Ошибка в analyze_sentiment: {e}")
            return self._fallback_sentiment_analysis(text)

    def _fallback_sentiment_analysis(self, text: str) -> float:
        """Резервный анализ тональности на ключевых словах"""
        positive_words = ["хорош", "прекрасн", "рад", "счастлив", "люблю", "отлично", "замечательно"]
        negative_words = ["плох", "ужасн", "грустн", "злюсь", "ненавижу", "печаль", "отстой"]
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        total = max(1, pos_count + neg_count)
        return (pos_count - neg_count) / total

    def get_state(self) -> Dict[str, Any]:
        try:
            mood_value = self.long_term_mood
            if mood_value > 0.6:
                mood_desc = "😊 Отличное"
            elif mood_value > 0.3:
                mood_desc = "🙂 Хорошее"
            elif mood_value < -0.6:
                mood_desc = "😭 Ужасное"
            elif mood_value < -0.3:
                mood_desc = "😢 Плохое"
            else:
                mood_desc = "😐 Нейтральное"
                
            return {
                "current_emotion": self.current_emotion,
                "emotion_icon": EMOTION_ENGINE_CONFIG["base_emotions"][self.current_emotion]["icon"],
                "intensity": round(self.emotion_intensity, 2),
                "long_term_mood": round(self.long_term_mood, 2),
                "mood_description": mood_desc,
                "triggers": list(self.triggers_activated),
                "history_size": len(self.emotion_history)
            }
        except Exception as e:
            logger.error(f"Ошибка в get_state: {e}")
            return {
                "current_emotion": "neutral",
                "emotion_icon": "😐",
                "intensity": 0.5,
                "long_term_mood": 0.0,
                "mood_description": "😐 Нейтральное",
                "triggers": [],
                "history_size": 0
            }

# --- КЛАСС ДОЛГОВРЕМЕННОЙ ПАМЯТИ ---
class LongTermMemory:
    def __init__(self):
        self.memory_file = MEMORY_DIR / "long_term_memory.json"
        self.memories = []
        self.sentence_model = None
        self._initialize_model()
        self._load_memory()

    def _initialize_model(self) -> None:
        try:
            sentence_model_name = "all-MiniLM-L6-v2"
            sentence_model_path = Path(sentence_model_name)
            
            if sentence_model_path.exists() and sentence_model_path.is_dir():
                model_source = str(sentence_model_path)
            else:
                model_source = sentence_model_name
                
            self.sentence_model = SentenceTransformer(
                model_source,
                device=device
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации модели эмбеддингов: {e}")
            self.sentence_model = None

    def _load_memory(self) -> None:
        if not self.memory_file.exists():
            return
            
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "memories" in data and isinstance(data["memories"], list):
                    self.memories = data["memories"]
                    logger.info(f"Загружено {len(self.memories)} воспоминаний из памяти")
                else:
                    logger.warning("Неверный формат файла памяти")
        except json.JSONDecodeError:
            logger.error("Ошибка декодирования JSON файла памяти")
        except Exception as e:
            logger.error(f"Ошибка загрузки памяти: {e}")

    def add_memory(self, text: str, emotion_state: Dict) -> None:
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            logger.warning("Попытка добавить пустое или слишком короткое воспоминание")
            return
            
        if self.sentence_model is None:
            logger.error("Модель для эмбеддингов не загружена")
            return
            
        try:
            # Ограничиваем длину текста для эмбеддинга
            text_processed = text[:512]
            embedding = self.sentence_model.encode(text_processed)
            
            memory = {
                "text": text_processed,
                "timestamp": datetime.now().isoformat(),
                "emotion": emotion_state,
                "embedding": embedding.tolist()
            }
            
            self.memories.append(memory)
            
            # Ограничиваем размер памяти
            if len(self.memories) > EMOTION_ENGINE_CONFIG["max_memory"]:
                self.memories = self.memories[-EMOTION_ENGINE_CONFIG["max_memory"]:]
                
            self._save_memory()
        except Exception as e:
            logger.error(f"Ошибка добавления в память: {e}")

    def _save_memory(self) -> None:
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"memories": self.memories},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти: {e}")

    def find_related_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        if not query or not isinstance(query, str) or not self.memories or self.sentence_model is None:
            return []
            
        try:
            query_embedding = self.sentence_model.encode(query[:512])  # Ограничение длины запроса
            memory_embeddings = np.array([np.array(m["embedding"]) for m in self.memories])
            
            similarities = cosine_similarity(
                [query_embedding],
                memory_embeddings
            )[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.memories[i] for i in top_indices if similarities[i] > 0.3]
        except Exception as e:
            logger.error(f"Ошибка поиска в памяти: {e}")
            return []

    def clear_memory(self) -> None:
        self.memories = []
        try:
            if self.memory_file.exists():
                self.memory_file.unlink()
        except Exception as e:
            logger.error(f"Ошибка очистки памяти: {e}")

# --- КЛАСС ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ ---
class QADataset(Dataset):
    def __init__(self, qa_pairs: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 256):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Валидация данных
        self._validate_data()

    def _validate_data(self) -> None:
        if not isinstance(self.qa_pairs, list):
            raise ValueError("qa_pairs должен быть списком")
            
        valid_pairs = []
        for pair in self.qa_pairs:
            if not isinstance(pair, dict):
                continue
            if "question" not in pair or "answer" not in pair:
                continue
            if not isinstance(pair["question"], str) or not isinstance(pair["answer"], str):
                continue
            if len(pair["question"]) < 3 or len(pair["answer"]) < 3:
                continue
            valid_pairs.append(pair)
            
        self.qa_pairs = valid_pairs
        if not self.qa_pairs:
            raise ValueError("Нет допустимых пар вопрос-ответ в датасете")

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.qa_pairs[idx]
        question = item['question']
        answer = item['answer']
        
        try:
            model_inputs = self.tokenizer(
                question, 
                max_length=self.max_length, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            labels = self.tokenizer(
                text_target=answer,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"]
            
            model_inputs = {key: val.squeeze(0) for key, val in model_inputs.items()}
            labels = labels.squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
            
            return model_inputs
        except Exception as e:
            logger.error(f"Ошибка обработки пары {idx}: {e}")
            # Возвращаем пустые тензоры в случае ошибки
            empty_tensor = torch.zeros((self.max_length,), dtype=torch.long)
            return {
                "input_ids": empty_tensor,
                "attention_mask": empty_tensor,
                "labels": empty_tensor.fill_(-100)
            }

# --- ОСНОВНОЙ КЛАСС БОТА ---
class EmotionalChatBot:
    def __init__(self, auto_load_model: bool = True):
        self.tokenizer = None
        self.model = None
        self.emotion_engine = EmotionalState()
        self.memory = LongTermMemory()
        self.conversation_history = []
        self.config = {
            "personality_traits": {"openness": 0.7, "agreeableness": 0.8},
            "last_trained": None,
            "version": "1.2",
            "created_at": datetime.now().isoformat()
        }
        
        self._initialize_model(auto_load_model)
        self._load_config()

    def _initialize_model(self, auto_load: bool) -> None:
        try:
            # Попытка загрузки локальной модели
            if (MODEL_DIR / "config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_DIR,
                    local_files_only=True,
                    use_fast=False,
                    legacy=False
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_DIR,
                    local_files_only=True,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                logger.info("Модель загружена из локального хранилища")
                return
                
            # Автозагрузка с Hugging Face
            if auto_load:
                model_name = "cointegrated/rut5-base"
                logger.info(f"Попытка загрузки модели {model_name} с Hugging Face")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                    device_map="auto"
                )
                
                # Сохраняем загруженную модель локально
                self.save_state()
                logger.info(f"Модель {model_name} успешно загружена и сохранена локально")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}\n{traceback.format_exc()}")
            self.tokenizer = None
            self.model = None

    def _load_config(self) -> None:
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    if isinstance(loaded_config, dict):
                        # Обновляем только существующие ключи
                        for key in self.config:
                            if key in loaded_config:
                                self.config[key] = loaded_config[key]
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")

    def generate_response(self, user_input: str) -> str:
        if not isinstance(user_input, str) or not user_input.strip():
            return "Пожалуйста, введите осмысленный текст"
            
        try:
            # Обновляем эмоциональное состояние
            self.emotion_engine.update(user_input)
            emotion_state = self.emotion_engine.get_state()
            
            # Добавляем в память
            self.memory.add_memory(user_input, emotion_state)
            
            # Если модель не загружена, возвращаем случайный ответ
            if self.model is None or self.tokenizer is None:
                return random.choice([
                    "Я пока не умею отвечать. Пожалуйста, обучите меня!",
                    "Моя модель не загружена. Выберите обучение в меню.",
                    "Я не могу сгенерировать ответ. Попробуйте позже.",
                    "Извините, я ещё не обучен. Хотите настроить меня?"
                ])
                
            # Поиск связанных воспоминаний
            related_memories = self.memory.find_related_memories(user_input, top_k=2)
            context = "\n".join([f"Ранее: {m['text']}" for m in related_memories]) if related_memories else ""
            
            # Формируем промпт с учетом эмоций и контекста
            prompt = f"""
Ты — Sin, эмоциональный ИИ-ассистент. Учитывай контекст и эмоции.
Текущее состояние: {emotion_state['emotion_icon']} {emotion_state['current_emotion']} 
Настроение: {emotion_state['mood_description']}
{context}
Пользователь: {user_input}
Sin:
""".strip()
            
            # Генерация ответа
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7 + emotion_state["intensity"] * 0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Декодируем и очищаем ответ
            full_response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()
            
            # Удаляем повтор prompt'а из ответа
            response = full_response[len(prompt):].strip() if full_response.startswith(prompt) else full_response
            
            # Обработка пустого ответа
            if not response:
                response = random.choice([
                    "Я вас услышал.",
                    "Интересный вопрос.",
                    "Дайте мне подумать...",
                    "Понял вас."
                ])
                
            # Добавляем эмоциональную иконку
            return f"{emotion_state['emotion_icon']} {response}"
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "⚠️ Не хватает памяти GPU. Попробуйте более короткий запрос."
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}\n{traceback.format_exc()}")
            return "Произошла ошибка при генерации ответа. Попробуйте снова."

    def save_state(self) -> None:
        try:
            # Сохраняем конфигурацию
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                
            # Сохраняем модель, если она есть
            if self.model is not None and self.tokenizer is not None:
                self.model.save_pretrained(MODEL_DIR)
                self.tokenizer.save_pretrained(MODEL_DIR)
                logger.info("Модель успешно сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}\n{traceback.format_exc()}")

    def clear_conversation_history(self) -> None:
        self.conversation_history = []
        logger.info("История диалога очищена")

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ ---
def load_text_from_file(file_path: Path) -> str:
    """Загружает текст из файла с автоматическим определением формата."""
    if not file_path.exists():
        logger.error(f"Файл не существует: {file_path}")
        return ""
        
    try:
        # TXT файлы
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        # PDF файлы
        elif file_path.suffix.lower() == '.pdf' and HAS_PYPDF:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        # DOCX файлы
        elif file_path.suffix.lower() == '.docx' and HAS_DOCX:
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
            
        else:
            logger.error(f"Неподдерживаемый формат файла: {file_path.suffix}")
            return ""
    except Exception as e:
        logger.error(f"Ошибка чтения файла {file_path}: {e}")
        return ""

def load_qa_pairs_from_text(text: str, max_pairs: int = 100) -> List[Dict[str, str]]:
    """Создает пары вопрос-ответ из текста."""
    if not text:
        return []
        
    try:
        # Разделяем текст на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if 10 < len(s.strip()) < 500]
        
        qa_pairs = []
        for i in range(len(sentences) - 1):
            if len(qa_pairs) >= max_pairs:
                break
                
            statement = sentences[i]
            next_statement = sentences[i+1]
            
            # Формируем вопрос из текущего предложения
            if statement.endswith('.'):
                question = statement[:-1] + '?'
            else:
                question = statement + '?'
                
            # Ответ - следующее предложение
            answer = next_statement
            
            qa_pairs.append({
                "question": question,
                "answer": answer
            })
            
        return qa_pairs
    except Exception as e:
        logger.error(f"Ошибка создания QA пар: {e}")
        return []

def load_dataset_from_files(data_dir: Path) -> List[Dict[str, str]]:
    """Загружает датасет из всех поддерживаемых файлов в директории."""
    if not data_dir.exists():
        logger.error(f"Директория не существует: {data_dir}")
        return []
        
    qa_pairs = []
    supported_extensions = ['.txt', '.pdf', '.docx']
    
    try:
        # Собираем все поддерживаемые файлы
        files = []
        for ext in supported_extensions:
            files.extend(list(data_dir.glob(f"*{ext}")))
            
        if not files:
            logger.warning(f"Нет поддерживаемых файлов в {data_dir}")
            return []
            
        # Обрабатываем каждый файл
        for file_path in files:
            logger.info(f"Обработка файла: {file_path.name}")
            
            text = load_text_from_file(file_path)
            if not text:
                continue
                
            # Создаем QA пары из текста
            pairs = load_qa_pairs_from_text(text, max_pairs=100)
            qa_pairs.extend(pairs)
            
            if len(qa_pairs) >= 500:  # Ограничение на общее количество пар
                break
                
        logger.info(f"Загружено {len(qa_pairs)} QA пар из файлов")
        return qa_pairs
    except Exception as e:
        logger.error(f"Ошибка загрузки датасета из файлов: {e}")
        return []

# --- ФУНКЦИИ ОБУЧЕНИЯ ---
def train_pytorch_model(
    bot: EmotionalChatBot,
    train_dataset: Dataset,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    save_steps: int = 10
) -> None:
    """Обучение модели с использованием PyTorch напрямую."""
    if bot.model is None or bot.tokenizer is None:
        logger.error("Модель или токенизатор не загружены")
        return
        
    try:
        # Настройка LoRA
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        if not isinstance(bot.model, PeftModel):
            model = get_peft_model(bot.model, peft_config)
        else:
            model = bot.model
            
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"Ошибка настройки LoRA: {e}")
        return

    try:
        # Подготовка DataLoader
        data_collator = DataCollatorForSeq2Seq(
            bot.tokenizer, 
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
    except Exception as e:
        logger.error(f"Ошибка подготовки данных: {e}")
        return

    # Настройка оптимизатора и планировщика
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps
    )

    model.to(device)
    model.train()
    
    logger.info(f"Начало обучения на {len(train_dataset)} примерах, {epochs} эпох")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Обучение...", total=total_steps)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in train_dataloader:
                    try:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        # Прямой проход
                        outputs = model(**batch)
                        loss = outputs.loss
                        
                        # Обратный проход
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        progress.update(
                            task,
                            description=f"[green]Эпоха {epoch+1}/{epochs}[/green] - Loss: {loss.item():.4f}"
                        )
                        progress.advance(task)
                        
                        # Промежуточное сохранение
                        if num_batches % save_steps == 0:
                            logger.info(f"Промежуточное сохранение на шаге {num_batches}")
                            adapter_dir = MODEL_DIR / f"adapter_step_{num_batches}"
                            os.makedirs(adapter_dir, exist_ok=True)
                            model.save_pretrained(adapter_dir)
                            bot.tokenizer.save_pretrained(adapter_dir)
                            
                    except Exception as e:
                        logger.error(f"Ошибка на шаге обучения: {e}")
                        continue
                        
                # Логирование после эпохи
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                logger.info(f"Эпоха {epoch+1} завершена. Средний Loss: {avg_loss:.4f}")
                
                # Очистка памяти
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        # Финальное сохранение
        logger.info("Сохранение обученной модели...")
        model.save_pretrained(MODEL_DIR)
        bot.tokenizer.save_pretrained(MODEL_DIR)
        bot.config["last_trained"] = datetime.now().isoformat()
        bot.save_state()
        
        logger.info("Обучение успешно завершено")
        
    except Exception as e:
        logger.error(f"Критическая ошибка обучения: {e}")

def train_with_sft(
    bot: EmotionalChatBot,
    train_data: List[Dict[str, str]],
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
) -> None:
    """Обучение с использованием Seq2SeqTrainer."""
    if bot.model is None or bot.tokenizer is None:
        logger.error("Модель или токенизатор не загружены")
        return
        
    try:
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_list(train_data)
        
        def preprocess_function(examples):
            inputs = [q for q in examples["question"]]
            targets = [a for a in examples["answer"]]
            
            model_inputs = bot.tokenizer(
                inputs,
                max_length=256,
                truncation=True,
                padding="max_length"
            )
            
            labels = bot.tokenizer(
                text_target=targets,
                max_length=256,
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
            
        tokenized_dataset = hf_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=hf_dataset.column_names
        )
    except Exception as e:
        logger.error(f"Ошибка подготовки данных: {e}")
        return

    try:
        # Настройка LoRA
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        model = get_peft_model(bot.model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"Ошибка настройки LoRA: {e}")
        return

    try:
        # Аргументы обучения
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(LOGS_DIR),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=None,
            remove_unused_columns=True,
            logging_first_step=True,
            predict_with_generate=True
        )
        
        # Data Collator
        data_collator = DataCollatorForSeq2Seq(
            bot.tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=bot.tokenizer,
            data_collator=data_collator
        )
        
        # Запуск обучения
        logger.info("Запуск обучения...")
        trainer.train()
        
        # Сохранение результатов
        logger.info("Сохранение модели...")
        model.save_pretrained(MODEL_DIR)
        bot.tokenizer.save_pretrained(MODEL_DIR)
        bot.config["last_trained"] = datetime.now().isoformat()
        bot.save_state()
        
        logger.info("Обучение успешно завершено")
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")

# --- RLHF ФУНКЦИИ ---
def collect_rlhf_feedback(bot: EmotionalChatBot) -> None:
    """Сбор оценок ответов для обучения с подкреплением."""
    try:
        if bot.model is None:
            console.print("[red]Ошибка: модель не загружена. Сначала обучите Sin![/red]")
            return

        feedback_file = DATA_DIR / "feedback.jsonl"
        console.print(Panel("🧠 Оцените ответы Sin (1–5)", style="bold yellow"))
        
        # Загрузка вопросов
        questions = []
        try:
            if RLHF_QUESTIONS_FILE.exists():
                with open(RLHF_QUESTIONS_FILE, "r", encoding="utf-8") as f:
                    questions = [q.strip() for q in f.readlines() if q.strip()]
        except Exception as e:
            logger.error(f"Ошибка чтения файла вопросов: {e}")
        
        if not questions:
            questions = [
                "Привет! Как дела?",
                "Что ты думаешь об искусственном интеллекте?",
                "Расскажи что-нибудь интересное",
                "Какой твой любимый фильм?",
                "Что ты умеешь?",
                "Как работает машинное обучение?",
                "Что такое трансформеры в NLP?",
                "Какой самый интересный факт о космосе?"
            ]
            logger.info("Используются стандартные вопросы для RLHF")

        feedback = []
        for q in random.sample(questions, min(5, len(questions))):
            try:
                response = bot.generate_response(q)
                console.print(f"[cyan]Вопрос:[/cyan] {q}")
                console.print(f"[magenta]Sin:[/magenta] {response}")
                
                while True:
                    rating = console.input("Оценка (1-5, или 0 для пропуска): ").strip()
                    if rating.isdigit() and 0 <= int(rating) <= 5:
                        if int(rating) > 0:
                            feedback.append({
                                "input": q,
                                "output": response,
                                "score": int(rating),
                                "timestamp": datetime.now().isoformat()
                            })
                        break
                    else:
                        console.print("[red]Введите число от 0 до 5.[/red]")
            except Exception as e:
                logger.error(f"Ошибка при обработке вопроса '{q}': {e}")
                continue

        if feedback:
            try:
                with open(feedback_file, "a", encoding="utf-8") as f:
                    for item in feedback:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                console.print(f"[green]Сохранено {len(feedback)} оценок.[/green]")
            except Exception as e:
                console.print(f"[red]Ошибка сохранения оценок: {e}[/red]")
                logger.exception("Ошибка сохранения RLHF feedback")
        else:
            console.print("[yellow]Оценки не собраны.[/yellow]")
            
    except Exception as e:
        logger.error(f"Критическая ошибка в collect_rlhf_feedback: {e}")
        console.print("[red]Произошла критическая ошибка при сборе оценок.[/red]")

# --- ГЛАВНОЕ МЕНЮ ---
def main_menu(bot: EmotionalChatBot) -> None:
    """Основное меню взаимодействия."""
    while True:
        console.print(Panel.fit("🤖 [bold green]Sin — эмоциональный ИИ-ассистент[/bold green]"))
        
        table = Table(title="Главное меню", show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim", width=4)
        table.add_column("Опция", width=40)
        
        menu_options = [
            ("1", "Чат с Sin"),
            ("2", "Сбор RLHF оценок"),
            ("3", "Обучение на стандартных данных"),
            ("4", "Обучение на данных из файлов"),
            ("5", "Обучение PyTorch (продвинутое)"),
            ("6", "Показать состояние"),
            ("7", "Очистить историю"),
            ("8", "Сохранить модель"),
            ("9", "Выход")
        ]
        
        for num, opt in menu_options:
            table.add_row(num, opt)
            
        console.print(table)
        
        choice = console.input("[bold]Выберите действие: [/bold]").strip()
        
        if choice == "1":
            chat_with_bot(bot)
        elif choice == "2":
            collect_rlhf_feedback(bot)
        elif choice == "3":
            train_with_default_data(bot)
        elif choice == "4":
            train_with_file_data(bot)
        elif choice == "5":
            train_with_pytorch(bot)
        elif choice == "6":
            show_bot_state(bot)
        elif choice == "7":
            bot.clear_conversation_history()
            console.print("[green]История диалога очищена[/green]")
        elif choice == "8":
            bot.save_state()
            console.print("[green]Модель сохранена[/green]")
        elif choice == "9":
            console.print("[bold red]Завершение работы...[/bold red]")
            break
        else:
            console.print("[red]Неверный выбор. Попробуйте снова.[/red]")

def chat_with_bot(bot: EmotionalChatBot) -> None:
    """Режим чата с ботом."""
    console.print(Panel.fit("💬 [bold blue]Режим чата (введите 'выход' для возврата)[/bold blue]"))
    
    while True:
        try:
            user_input = console.input("[bold cyan]Вы: [/bold cyan]").strip()
            if user_input.lower() in ("выход", "exit", "quit"):
                break
                
            if not user_input:
                console.print("[yellow]Пожалуйста, введите сообщение[/yellow]")
                continue
                
            # Генерация ответа
            response = bot.generate_response(user_input)
            console.print(f"[bold magenta]Sin:[/bold magenta] {response}")
            
            # Показ состояния
            emotion_state = bot.emotion_engine.get_state()
            state_table = Table(title="Состояние Sin", show_header=False)
            state_table.add_row("Эмоция", f"{emotion_state['emotion_icon']} {emotion_state['current_emotion']}")
            state_table.add_row("Интенсивность", f"{emotion_state['intensity']:.2f}")
            state_table.add_row("Настроение", emotion_state['mood_description'])
            console.print(state_table)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Ошибка в чате: {e}")
            console.print("[red]Произошла ошибка. Попробуйте снова.[/red]")

def train_with_default_data(bot: EmotionalChatBot) -> None:
    """Обучение на стандартном наборе данных."""
    console.print(Panel.fit("📚 [bold blue]Обучение на стандартных данных[/bold blue]"))
    
    default_data = [
        {"question": "Привет", "answer": "Здравствуй! Как твои дела?"},
        {"question": "Как дела?", "answer": "У меня всё хорошо, спасибо! А у тебя?"},
        {"question": "Что ты умеешь?", "answer": "Я могу общаться на разные темы и поддерживать беседу."},
        {"question": "Расскажи анекдот", "answer": "Программист звонит в библиотеку и спрашивает: 'У вас есть книги про ООП?' Библиотекарь отвечает: 'Да, но они не возвращаемые!'"},
        {"question": "Как тебя зовут?", "answer": "Меня зовут Sin. Я твой виртуальный помощник!"},
        {"question": "Пока", "answer": "До свидания! Возвращайся скорее!"},
        {"question": "Спасибо", "answer": "Пожалуйста! Рад был помочь."},
        {"question": "Что такое ИИ?", "answer": "Искусственный интеллект — это область компьютерных наук, которая создает системы, способные выполнять задачи, требующие человеческого интеллекта."}
    ]
    
    try:
        console.print("[blue]Начало обучения...[/blue]")
        train_with_sft(bot, default_data, epochs=3)
        console.print("[green]Обучение завершено успешно![/green]")
    except Exception as e:
        console.print(f"[red]Ошибка обучения: {e}[/red]")
        logger.error(f"Ошибка обучения на стандартных данных: {e}")

def train_with_file_data(bot: EmotionalChatBot) -> None:
    """Обучение на данных из файлов."""
    console.print(Panel.fit("📂 [bold blue]Обучение на данных из файлов[/bold blue]"))
    
    if not DATA_DIR.exists():
        console.print(f"[red]Директория с данными не найдена: {DATA_DIR}[/red]")
        return
        
    console.print(f"[blue]Загрузка данных из {DATA_DIR}...[/blue]")
    qa_pairs = load_dataset_from_files(DATA_DIR)
    
    if not qa_pairs:
        console.print("[red]Не удалось загрузить данные для обучения[/red]")
        return
        
    try:
        console.print(f"[blue]Начало обучения на {len(qa_pairs)} примерах...[/blue]")
        train_with_sft(bot, qa_pairs, epochs=2)
        console.print("[green]Обучение завершено успешно![/green]")
    except Exception as e:
        console.print(f"[red]Ошибка обучения: {e}[/red]")
        logger.error(f"Ошибка обучения на данных из файлов: {e}")

def train_with_pytorch(bot: EmotionalChatBot) -> None:
    """Продвинутое обучение с использованием PyTorch."""
    console.print(Panel.fit("🧠 [bold blue]Продвинутое обучение PyTorch[/bold blue]"))
    
    if not DATA_DIR.exists():
        console.print(f"[red]Директория с данными не найдена: {DATA_DIR}[/red]")
        return
        
    console.print(f"[blue]Загрузка данных из {DATA_DIR}...[/blue]")
    qa_pairs = load_dataset_from_files(DATA_DIR)
    
    if not qa_pairs:
        console.print("[red]Не удалось загрузить данные для обучения[/red]")
        return
        
    if bot.tokenizer is None:
        console.print("[red]Токенизатор не загружен[/red]")
        return
        
    try:
        console.print("[blue]Создание датасета...[/blue]")
        dataset = QADataset(qa_pairs, bot.tokenizer)
        
        console.print(f"[blue]Начало обучения на {len(dataset)} примерах...[/blue]")
        train_pytorch_model(
            bot,
            dataset,
            epochs=2,
            batch_size=1,
            learning_rate=3e-5,
            save_steps=20
        )
        console.print("[green]Обучение завершено успешно![/green]")
    except Exception as e:
        console.print(f"[red]Ошибка обучения: {e}[/red]")
        logger.error(f"Ошибка PyTorch обучения: {e}")

def show_bot_state(bot: EmotionalChatBot) -> None:
    """Показывает текущее состояние бота."""
    state = bot.emotion_engine.get_state()
    memory_size = len(bot.memory.memories)
    model_status = "Загружена" if bot.model is not None else "Не загружена"
    
    table = Table(title="Состояние Sin", show_header=True, header_style="bold blue")
    table.add_column("Параметр")
    table.add_column("Значение")
    
    table.add_row("Модель", model_status)
    table.add_row("Эмоция", f"{state['emotion_icon']} {state['current_emotion']}")
    table.add_row("Интенсивность", str(state['intensity']))
    table.add_row("Настроение", state['mood_description'])
    table.add_row("Воспоминания", str(memory_size))
    table.add_row("Последнее обучение", bot.config.get("last_trained", "Никогда"))
    
    console.print(table)

# --- ТОЧКА ВХОДА ---
def main():
    try:
        console.print(Panel.fit("🤖 [bold green]Загрузка Sin...[/bold green]"))
        
        # Инициализация бота
        bot = EmotionalChatBot(auto_load_model=True)
        
        # Проверка загрузки модели
        if bot.model is None:
            console.print("[yellow]Модель не загружена. Используйте обучение в меню.[/yellow]")
        
        # Запуск главного меню
        main_menu(bot)
        
    except KeyboardInterrupt:
        console.print("\n[red]Завершение работы по запросу пользователя...[/red]")
    except Exception as e:
        console.print(f"[red]Критическая ошибка: {e}[/red]")
        logger.critical(f"Критическая ошибка: {e}\n{traceback.format_exc()}")
    finally:
        console.print("[bold blue]Sin завершает работу. До свидания![/bold blue]")

if __name__ == "__main__":
    main()
