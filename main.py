import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
import os
import re
import glob
import time
from datetime import datetime
from collections import Counter
import logging
from torch.nn.utils.rnn import pad_sequence

# Для работы с DOCX файлами
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️  python-docx не установлен. Установите его для поддержки .docx файлов: pip install python-docx")

# Для работы с PDF файлами
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyPDF2 не установлен. Установите его для поддержки .pdf файлов: pip install PyPDF2")

# ------------------
# Настройка логирования
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------
# Гиперпараметры
# ------------------
DEFAULT_SEQ_LENGTH = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_NUM_LAYERS = 3
DEFAULT_DROPOUT = 0.3
DEFAULT_WARMUP_STEPS = 1000
MAX_SAVED_MODELS = 3
DEFAULT_TOKEN_TYPE = "char"  # "char" или "word"

# ------------------
# Продвинутые техники сэмплирования
# ------------------
def advanced_sampling(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Продвинутое сэмплирование с temperature, top-k и top-p"""
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Собираем индексы для удаления
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(sorted_indices.size(0)):
            indices_to_remove[i, sorted_indices[i, sorted_indices_to_remove[i]]] = True
        logits[indices_to_remove] = float('-inf')
    
    return F.softmax(logits, dim=-1)

# ------------------
# Early Stopping
# ------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ------------------
# Label Smoothing Loss
# ------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ------------------
# Подготовка данных
# ------------------
def load_text(file_path):
    """Загрузка текста из файла различных форматов"""
    logger.info(f"Попытка загрузки файла: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.txt':
            return load_txt_file(file_path)
        elif file_extension == '.docx':
            return load_docx_file(file_path)
        elif file_extension == '.pdf':
            return load_pdf_file(file_path)
        else:
            # Попытка загрузить как текстовый файл
            logger.warning(f"Неизвестный формат файла {file_extension}, пробуем загрузить как текст")
            return load_txt_file(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
        raise

def load_txt_file(file_path):
    """Загрузка текстового файла"""
    encodings = ['utf-8', 'windows-1251', 'cp1251', 'koi8-r', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            logger.info(f"Файл {file_path} успешно загружен с кодировкой {encoding}")
            return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path} с кодировкой {encoding}: {e}")
            continue
    
    # Если все кодировки не сработали, попробуем игнорировать ошибки
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logger.warning(f"Файл {file_path} загружен с игнорированием ошибок кодировки")
        return text
    except Exception as e:
        raise Exception(f"Не удалось загрузить файл {file_path} ни с одной кодировкой: {e}")

def load_docx_file(file_path):
    """Загрузка DOCX файла"""
    if not DOCX_SUPPORT:
        raise Exception("Поддержка DOCX файлов не доступна. Установите python-docx")
    
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        logger.info(f"DOCX файл {file_path} успешно загружен")
        return text
    except Exception as e:
        raise Exception(f"Ошибка при загрузке DOCX файла {file_path}: {e}")

def load_pdf_file(file_path):
    """Загрузка PDF файла"""
    if not PDF_SUPPORT:
        raise Exception("Поддержка PDF файлов не доступна. Установите PyPDF2")
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        logger.info(f"PDF файл {file_path} успешно загружен")
        return text
    except Exception as e:
        raise Exception(f"Ошибка при загрузке PDF файла {file_path}: {e}")

def tokenize_text(text, token_type="char"):
    """Токенизация текста"""
    if token_type == "char":
        return list(text)
    elif token_type == "word":
        # Простая токенизация по словам
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return words
    else:
        raise ValueError(f"Неизвестный тип токенизации: {token_type}")

def clean_text(text):
    """Очистка текста от лишних символов"""
    original_length = len(text)
    # Оставляем буквы, цифры, пробелы и основную пунктуацию
    text = re.sub(r'[^\w\s\.\,\!\?\-\n\u0400-\u04FF]', '', text)  # Добавлена поддержка кириллицы
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    # Удаляем множественные переводы строк
    text = re.sub(r'\n+', '\n', text)
    cleaned_length = len(text)
    logger.info(f"Текст очищен: {original_length} -> {cleaned_length} символов")
    return text.strip()

def create_token_mappings(tokens):
    """Создание словарей для кодирования/декодирования токенов"""
    unique_tokens = sorted(list(set(tokens)))
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    logger.info(f"Созданы словари: {len(unique_tokens)} уникальных токенов")
    return token_to_idx, idx_to_token, len(unique_tokens)

def create_sequences(tokens, token_to_idx, seq_length):
    """Создание последовательностей для обучения"""
    if len(tokens) < seq_length + 1:
        raise ValueError(f"Текст слишком короткий. Минимальная длина: {seq_length + 1}, текущая: {len(tokens)}")
    
    data = [token_to_idx[token] for token in tokens]
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    logger.info(f"Создано {len(X)} последовательностей длиной {seq_length}")
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ------------------
# Улучшенная модель с Attention
# ------------------
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        # Attention механизм
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Простое attention - средневзвешенное по времени
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output, hidden

    def get_model_info(self):
        """Получение информации о модели"""
        info = f"AttentionRNN Model:\n"
        info += f"  Vocabulary size: {self.vocab_size}\n"
        info += f"  Hidden size: {self.hidden_size}\n"
        info += f"  Number of layers: {self.num_layers}\n"
        info += f"  Parameters: {sum(p.numel() for p in self.parameters()):,}"
        return info

# ------------------
# Управление моделями
# ------------------
def get_model_files():
    """Получение списка сохраненных моделей"""
    model_files = glob.glob("char_rnn_model_*.pth")
    model_files.sort(key=os.path.getctime, reverse=True)  # Сортировка по времени создания
    return model_files

def cleanup_old_models():
    """Удаление старых моделей если их больше MAX_SAVED_MODELS"""
    model_files = get_model_files()
    if len(model_files) > MAX_SAVED_MODELS:
        old_models = model_files[MAX_SAVED_MODELS:]
        for old_model in old_models:
            try:
                os.remove(old_model)
                logger.info(f"Удалена старая модель: {old_model}")
            except Exception as e:
                logger.error(f"Ошибка при удалении модели {old_model}: {e}")

def save_model_with_timestamp(model, token_to_idx, idx_to_token, vocab_size, 
                            loss=0.0, token_type="char", perplexity=None):
    """Сохранение модели с таймстампом"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"char_rnn_model_{timestamp}.pth"
    
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'vocab_size': vocab_size,
            'timestamp': timestamp,
            'loss': loss,
            'perplexity': perplexity,
            'token_type': token_type,
            'model_config': {
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout': model.dropout.p,
                'model_type': type(model).__name__
            }
        }, model_path)
        logger.info(f"Модель сохранена: {model_path}")
        
        # Очистка старых моделей
        cleanup_old_models()
        return model_path
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")
        return None

def load_model_with_dicts(model_path, device):
    """Загрузка модели с параметрами"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {})
        
        # Создание новой модели с теми же параметрами
        model_class = model_config.get('model_type', 'AttentionRNN')
        if model_class == 'AttentionRNN':
            model = AttentionRNN(
                checkpoint['vocab_size'],
                model_config.get('hidden_size', DEFAULT_HIDDEN_SIZE),
                model_config.get('num_layers', DEFAULT_NUM_LAYERS),
                model_config.get('dropout', DEFAULT_DROPOUT)
            )
        else:
            # По умолчанию
            model = AttentionRNN(
                checkpoint['vocab_size'],
                model_config.get('hidden_size', DEFAULT_HIDDEN_SIZE),
                model_config.get('num_layers', DEFAULT_NUM_LAYERS),
                model_config.get('dropout', DEFAULT_DROPOUT)
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        token_to_idx = checkpoint['token_to_idx']
        idx_to_token = checkpoint['idx_to_token']
        timestamp = checkpoint.get('timestamp', 'unknown')
        loss = checkpoint.get('loss', 0.0)
        token_type = checkpoint.get('token_type', 'char')
        perplexity = checkpoint.get('perplexity', None)
        
        logger.info(f"Модель загружена: {model_path} (timestamp: {timestamp}, loss: {loss:.4f})")
        return model, token_to_idx, idx_to_token, token_type, perplexity
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_path}: {e}")
        return None, None, None, None, None

def list_available_models():
    """Список доступных моделей"""
    model_files = get_model_files()
    if not model_files:
        print("Нет доступных моделей")
        return []
    
    print("\nДоступные модели:")
    for i, model_file in enumerate(model_files):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            timestamp = checkpoint.get('timestamp', 'unknown')
            loss = checkpoint.get('loss', 0.0)
            vocab_size = checkpoint.get('vocab_size', 0)
            token_type = checkpoint.get('token_type', 'char')
            perplexity = checkpoint.get('perplexity', 'N/A')
            print(f"{i+1}. {os.path.basename(model_file)}")
            print(f"   Дата: {timestamp}, Loss: {loss:.4f}, Perplexity: {perplexity}, Vocab: {vocab_size}, Type: {token_type}")
        except Exception as e:
            print(f"{i+1}. {os.path.basename(model_file)} (ошибка чтения: {e})")
    return model_files

# ------------------
# Расчет перплексии
# ------------------
def calculate_perplexity(model, data_loader, device, criterion):
    """Расчет перплексии модели"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output, _ = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    return perplexity

# ------------------
# Генерация текста
# ------------------
def generate_text(model, token_to_idx, idx_to_token, start_tokens, 
                 length=200, temperature=1.0, top_k=0, top_p=1.0, 
                 device='cpu', token_type="char"):
    """Генерация текста с продвинутыми методами сэмплирования"""
    logger.info(f"Начало генерации текста: '{start_tokens}', длина: {length}")
    logger.info(f"Параметры: температура={temperature}, top_k={top_k}, top_p={top_p}")
    
    model.eval()
    with torch.no_grad():
        tokens = tokenize_text(start_tokens, token_type) if token_type == "word" else list(start_tokens)
        
        # Проверка наличия всех токенов в словаре
        missing_tokens = [token for token in tokens if token not in token_to_idx]
        if missing_tokens:
            logger.warning(f"Токены отсутствуют в словаре: {missing_tokens}")
        
        input_seq = torch.tensor([token_to_idx.get(token, 0) for token in tokens], 
                               dtype=torch.long).unsqueeze(0).to(device)
        
        # Инициализация скрытого состояния
        hidden = None
        
        # Прогрев модели на начальных токенах
        for i in range(len(tokens) - 1):
            _, hidden = model(input_seq[:, i:i+1], hidden)
        
        # Генерация нового текста
        generated_tokens = tokens.copy()
        last_token_idx = token_to_idx.get(tokens[-1], 0)
        
        for _ in range(length):
            input_tensor = torch.tensor([[last_token_idx]], dtype=torch.long).to(device)
            output, hidden = model(input_tensor, hidden)
            
            # Продвинутое сэмплирование
            probs = advanced_sampling(output, temperature=temperature, top_k=top_k, top_p=top_p)
            try:
                top_i = torch.multinomial(probs, 1)[0]
                predicted_token = idx_to_token[top_i.item()]
                generated_tokens.append(predicted_token)
                last_token_idx = top_i.item()
            except Exception as e:
                logger.error(f"Ошибка при генерации токена: {e}")
                generated_tokens.append('?')
                last_token_idx = 0
            
        # Преобразование токенов в текст
        if token_type == "word":
            generated_text = ' '.join(generated_tokens)
        else:
            generated_text = ''.join(generated_tokens)
            
        logger.info(f"Генерация завершена, сгенерировано {len(generated_tokens)} токенов")
        return generated_text

# ------------------
# Обучение
# ------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
                token_to_idx, idx_to_token, vocab_size, token_type="char"):
    """Обучение модели с продвинутыми техниками"""
    logger.info(f"Начало обучения модели на устройстве {device}")
    logger.info(f"Параметры обучения: epochs={epochs}, batch_size={train_loader.batch_size}")
    
    model.to(device)
    model.train()
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_loss = float('inf')
    best_perplexity = float('inf')
    best_model_path = None
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        total_batches = len(train_loader)
        
        logger.info(f"Эпоха {epoch+1}/{epochs} начата")
        
        # Training phase
        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                output, _ = model(x_batch)
                loss = criterion(output, y_batch)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Логирование
            if batch_idx % max(1, total_batches // 10) == 0 and batch_idx > 0:
                avg_batch_loss = total_loss / (batch_idx + 1)
                logger.info(f"Эпоха {epoch+1}/{epochs}, Батч {batch_idx}/{total_batches}, Loss: {avg_batch_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output, _ = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, val_loader, device, criterion)
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / total_batches
        logger.info(f"Эпоха {epoch+1}/{epochs} завершена за {epoch_time:.2f} сек")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"  Perplexity: {perplexity:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping на эпохе {epoch+1}")
            break
        
        # Сохранение лучшей модели
        if val_loss < best_loss:
            best_loss = val_loss
            best_perplexity = perplexity
            # Сохранение модели с текущими параметрами
            model_path = save_model_with_timestamp(model, token_to_idx, idx_to_token, 
                                                 vocab_size, val_loss, token_type, perplexity)
            if model_path:
                best_model_path = model_path
                logger.info(f"Новая лучшая модель сохранена: loss {val_loss:.4f}, perplexity {perplexity:.4f}")
    
    training_time = time.time() - training_start_time
    logger.info(f"Обучение завершено за {training_time:.2f} сек")
    logger.info(f"Лучшая модель: loss {best_loss:.4f}, perplexity {best_perplexity:.4f}")
    
    return best_model_path

# ------------------
# Интерактивный режим
# ------------------
def interactive_mode():
    """Интерактивный режим работы с ИИ"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Запуск интерактивного режима на устройстве: {device}")
    
    current_model = None
    token_to_idx = None
    idx_to_token = None
    vocab_size = 0
    current_token_type = "char"
    current_perplexity = None
    
    print("\n" + "="*70)
    print("🤖 Улучшенный генеративный ИИ с_attention и продвинутыми техниками")
    print("="*70)
    print("Доступные команды:")
    print("  generate     - Генерация текста (продвинутая)")
    print("  train        - Обучение модели")
    print("  save         - Сохранение текущей модели")
    print("  load         - Загрузка модели")
    print("  list         - Список доступных моделей")
    print("  info         - Информация о текущей модели")
    print("  quit         - Выход")
    print("="*70)
    
    while True:
        try:
            command = input("\nВведите команду: ").strip().lower()
            
            if command == "quit":
                # Автоматическое сохранение при выходе
                if current_model is not None:
                    print("Автоматическое сохранение модели...")
                    save_model_with_timestamp(current_model, token_to_idx, idx_to_token, 
                                            vocab_size, token_type=current_token_type,
                                            perplexity=current_perplexity)
                print("До свидания!")
                break
                
            elif command == "generate":
                if current_model is None:
                    print("❌ Нет загруженной модели. Сначала загрузите или обучите модель.")
                    continue
                
                start_text = input("Введите начальный текст: ").strip()
                if not start_text:
                    start_text = "машинное" if current_token_type == "char" else "машинное обучение"
                
                try:
                    temp = float(input("Температура (0.1-2.0, по умолчанию 1.0): ") or "1.0")
                    temp = max(0.1, min(2.0, temp))
                except ValueError:
                    temp = 1.0
                    
                try:
                    top_k = int(input("Top-K (0 для отключения, по умолчанию 50): ") or "50")
                    top_k = max(0, top_k)
                except ValueError:
                    top_k = 50
                    
                try:
                    top_p = float(input("Top-P (0.0-1.0, по умолчанию 0.9): ") or "0.9")
                    top_p = max(0.0, min(1.0, top_p))
                except ValueError:
                    top_p = 0.9
                    
                try:
                    length = int(input("Длина текста (по умолчанию 200): ") or "200")
                    length = max(10, min(1000, length))
                except ValueError:
                    length = 200
                
                print("\n🔄 Генерация текста...")
                try:
                    generated = generate_text(current_model, token_to_idx, idx_to_token, 
                                            start_tokens=start_text, length=length,
                                            temperature=temp, top_k=top_k, top_p=top_p,
                                            device=device, token_type=current_token_type)
                    print(f"\n📝 Сгенерированный текст:\n{generated}")
                except Exception as e:
                    logger.error(f"Ошибка при генерации текста: {e}")
                    print(f"❌ Ошибка при генерации текста: {e}")
                
            elif command == "train":
                file_path = input("Введите путь к текстовому файлу: ").strip()
                if not file_path:
                    print("❌ Путь к файлу не указан")
                    continue
                
                if not os.path.exists(file_path):
                    print(f"❌ Файл {file_path} не найден")
                    continue
                
                try:
                    # Выбор типа токенизации
                    token_type = input("Тип токенизации (char/word, по умолчанию char): ").strip().lower()
                    if token_type not in ["char", "word"]:
                        token_type = "char"
                    
                    # Запрос параметров обучения
                    try:
                        epochs = int(input(f"Количество эпох (по умолчанию {DEFAULT_EPOCHS}): ") or str(DEFAULT_EPOCHS))
                    except ValueError:
                        epochs = DEFAULT_EPOCHS
                    
                    try:
                        seq_length = int(input(f"Длина последовательности (по умолчанию {DEFAULT_SEQ_LENGTH}): ") or str(DEFAULT_SEQ_LENGTH))
                    except ValueError:
                        seq_length = DEFAULT_SEQ_LENGTH
                    
                    try:
                        batch_size = int(input(f"Размер батча (по умолчанию {DEFAULT_BATCH_SIZE}): ") or str(DEFAULT_BATCH_SIZE))
                    except ValueError:
                        batch_size = DEFAULT_BATCH_SIZE
                    
                    # Загрузка и подготовка текста
                    print("🔄 Загрузка текста...")
                    text = load_text(file_path)
                    text = clean_text(text)
                    
                    # Токенизация
                    tokens = tokenize_text(text, token_type)
                    print(f"Текст токенизирован: {len(tokens)} токенов")
                    
                    if len(tokens) < seq_length * 2:
                        print("❌ Текст слишком короткий для обучения")
                        continue
                    
                    token_to_idx, idx_to_token, vocab_size = create_token_mappings(tokens)
                    X, y = create_sequences(tokens, token_to_idx, seq_length)
                    
                    if len(X) == 0:
                        print("❌ Недостаточно данных для обучения")
                        continue
                    
                    # Разделение на train/val
                    dataset = torch.utils.data.TensorDataset(X, y)
                    train_size = int(0.9 * len(dataset))
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                    
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                    # Создание модели
                    print("🔄 Создание модели...")
                    current_model = AttentionRNN(vocab_size, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT)
                    print(current_model.get_model_info())
                    
                    # Определение функции потерь и оптимизатора
                    criterion = LabelSmoothingLoss(smoothing=0.1)
                    optimizer = optim.Adam(current_model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=1e-5)
                    
                    current_token_type = token_type
                    
                    # Обучение модели
                    print("🔄 Начало обучения...")
                    model_path = train_model(current_model, train_loader, val_loader, criterion, optimizer, 
                                           epochs, device, token_to_idx, idx_to_token, vocab_size, token_type)
                    
                    if model_path:
                        print(f"✅ Обучение завершено! Модель сохранена в {os.path.basename(model_path)}")
                    else:
                        print("⚠️  Обучение завершено, но модель не была сохранена")
                        
                except Exception as e:
                    logger.error(f"Ошибка при обучении: {e}")
                    print(f"❌ Ошибка при обучении: {e}")
                    
            elif command == "save":
                if current_model is None:
                    print("❌ Нет модели для сохранения")
                    continue
                
                model_path = save_model_with_timestamp(current_model, token_to_idx, idx_to_token, 
                                                     vocab_size, token_type=current_token_type,
                                                     perplexity=current_perplexity)
                if model_path:
                    print(f"✅ Модель сохранена в {os.path.basename(model_path)}")
                else:
                    print("❌ Ошибка при сохранении модели")
                    
            elif command == "load":
                model_files = list_available_models()
                if not model_files:
                    continue
                
                try:
                    choice = int(input("Выберите модель (номер): ")) - 1
                    if 0 <= choice < len(model_files):
                        model_path = model_files[choice]
                        loaded_model, loaded_token_to_idx, loaded_idx_to_token, loaded_token_type, loaded_perplexity = load_model_with_dicts(model_path, device)
                        if loaded_model is not None:
                            current_model = loaded_model
                            token_to_idx = loaded_token_to_idx
                            idx_to_token = loaded_idx_to_token
                            current_token_type = loaded_token_type
                            current_perplexity = loaded_perplexity
                            
                            # Получение размера словаря из загруженной модели
                            checkpoint = torch.load(model_path, map_location=device)
                            vocab_size = checkpoint.get('vocab_size', len(token_to_idx))
                            
                            print(f"✅ Модель загружена из {os.path.basename(model_path)}")
                            print(f"   Тип токенизации: {current_token_type}")
                            if current_perplexity:
                                print(f"   Perplexity: {current_perplexity:.4f}")
                        else:
                            print("❌ Ошибка при загрузке модели")
                    else:
                        print("❌ Неверный номер модели")
                except ValueError:
                    print("❌ Неверный ввод")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке модели: {e}")
                    print(f"❌ Ошибка при загрузке модели: {e}")
                    
            elif command == "list":
                list_available_models()
                
            elif command == "info":
                if current_model is None:
                    print("❌ Нет загруженной модели")
                    continue
                
                print("\nИнформация о текущей модели:")
                print(current_model.get_model_info())
                print(f"  Vocabulary size (loaded): {vocab_size}")
                print(f"  Token type: {current_token_type}")
                print(f"  Device: {device}")
                if current_perplexity:
                    print(f"  Perplexity: {current_perplexity:.4f}")
                
                if token_to_idx and idx_to_token:
                    print(f"  Dictionary size: {len(token_to_idx)} tokens")
                    sample_tokens = list(token_to_idx.keys())[:30]
                    print(f"  Sample tokens: {sample_tokens}")
                
            else:
                print("❓ Неизвестная команда. Доступные команды: generate, train, save, load, list, info, quit")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Прерывание программы...")
            # Автоматическое сохранение при Ctrl+C
            if current_model is not None:
                print("Автоматическое сохранение модели...")
                save_model_with_timestamp(current_model, token_to_idx, idx_to_token, 
                                        vocab_size, token_type=current_token_type,
                                        perplexity=current_perplexity)
            print("До свидания!")
            break
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    print("🤖 Улучшенный генеративный ИИ с_attention и продвинутыми техниками")
    print("Поддерживаемые форматы файлов: .txt, .docx, .pdf")
    print("Новые возможности: attention, mixed precision, early stopping, advanced sampling")
    print("Запуск интерактивного режима...")
    interactive_mode()
