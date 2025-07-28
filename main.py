import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import re
from collections import Counter

# ------------------
# Гиперпараметры
# ------------------
SEQ_LENGTH = 25
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# ------------------
# Подготовка данных
# ------------------
def load_text(file_path):
    """Загрузка текста из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def clean_text(text):
    """Очистка текста от лишних символов"""
    # Оставляем буквы, цифры, пробелы и основную пунктуацию
    text = re.sub(r'[^\w\s\.\,\!\?\-\n]', '', text)
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def create_char_mappings(text):
    """Создание словарей для кодирования/декодирования символов"""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)

def create_sequences(text, char_to_idx, seq_length):
    """Создание последовательностей для обучения"""
    data = [char_to_idx[ch] for ch in text]
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ------------------
# Модель
# ------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        """Инициализация скрытого состояния"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)

# ------------------
# Генерация текста
# ------------------
def generate_text(model, char_to_idx, idx_to_char, start_str, length=200, temperature=1.0, device='cpu'):
    """Генерация текста с учетом температуры"""
    model.eval()
    with torch.no_grad():
        chars = list(start_str.lower())
        input_seq = torch.tensor([char_to_idx.get(ch, 0) for ch in chars], dtype=torch.long).unsqueeze(0).to(device)
        
        # Инициализация скрытого состояния
        hidden = model.init_hidden(1, device)
        
        # Прогрев модели на начальной строке
        for i in range(len(chars) - 1):
            _, hidden = model(input_seq[:, i:i+1], hidden)
        
        # Генерация нового текста
        last_char_idx = char_to_idx.get(chars[-1], 0)
        for _ in range(length):
            input_tensor = torch.tensor([[last_char_idx]], dtype=torch.long).to(device)
            output, hidden = model(input_tensor, hidden)
            
            # Применение температуры
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            predicted_char = idx_to_char[top_i.item()]
            chars.append(predicted_char)
            last_char_idx = top_i.item()
            
        return ''.join(chars)

# ------------------
# Обучение
# ------------------
def train_model(model, data_loader, criterion, optimizer, epochs, device, model_path="char_rnn.pth"):
    """Обучение модели"""
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = criterion(output.reshape(-1, output.size(-1)), y_batch)
            loss.backward()
            
            # Gradient clipping для стабильности обучения
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Сохранение модели после каждой эпохи
        save_model(model, model_path, epoch, avg_loss)
    
    print("Обучение завершено!")

def save_model(model, filepath, epoch, loss):
    """Сохранение модели"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Модель сохранена в {filepath}")

def load_model(model, filepath, device):
    """Загрузка модели"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Модель загружена: эпоха {epoch}, loss {loss:.4f}")
        return model
    else:
        print(f"Файл {filepath} не найден. Создается новая модель.")
        return model

# ------------------
# Основной код
# ------------------
def main():
    # Проверка доступности GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создание файла с примером текста (если его нет)
    sample_text_file = "sample_text.txt"
    if not os.path.exists(sample_text_file):
        sample_text = """
        Машинное обучение — это область искусственного интеллекта, 
        в которой компьютеры обучаются распознавать паттерны и 
        принимать решения без явного программирования. 
        Глубокое обучение использует нейронные сети с несколькими слоями 
        для анализа различных факторов данных. 
        Современные технологии позволяют создавать 
        интеллектуальные системы, способные к самообучению.
        """
        with open(sample_text_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"Создан пример текста в файле {sample_text_file}")
    
    # Загрузка и подготовка текста
    print("Загрузка текста...")
    text = load_text(sample_text_file)
    text = clean_text(text)
    print(f"Размер текста: {len(text)} символов")
    
    # Создание словарей
    char_to_idx, idx_to_char, vocab_size = create_char_mappings(text)
    print(f"Размер словаря: {vocab_size} символов")
    
    # Создание последовательностей
    print("Создание последовательностей...")
    X, y = create_sequences(text, char_to_idx, SEQ_LENGTH)
    print(f"Создано {len(X)} последовательностей")
    
    # Создание DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Создание модели
    model = CharRNN(vocab_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    
    # Загрузка существующей модели (если есть)
    model_path = "char_rnn_model.pth"
    model = load_model(model, model_path, device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение модели
    print("Начало обучения...")
    train_model(model, data_loader, criterion, optimizer, EPOCHS, device, model_path)
    
    # Генерация текста
    print("\n" + "="*50)
    print("ГЕНЕРАЦИЯ ТЕКСТА")
    print("="*50)
    
    start_texts = ["машинное", "искусственный", "обучение"]
    
    for start in start_texts:
        print(f"\nНачальный текст: '{start}'")
        generated = generate_text(model, char_to_idx, idx_to_char, 
                                start_str=start, length=200, temperature=0.8, device=device)
        print(f"Сгенерированный текст:\n{generated}")
        print("-" * 50)

# ------------------
# Функция для генерации текста из командной строки
# ------------------
def generate_from_cli():
    """Генерация текста из командной строки"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Проверка наличия модели
    model_path = "char_rnn_model.pth"
    if not os.path.exists(model_path):
        print("Модель не найдена. Сначала обучите модель.")
        return
    
    # Загрузка текста для создания словарей
    sample_text_file = "sample_text.txt"
    if not os.path.exists(sample_text_file):
        print("Файл с текстом не найден.")
        return
    
    text = load_text(sample_text_file)
    text = clean_text(text)
    char_to_idx, idx_to_char, vocab_size = create_char_mappings(text)
    
    # Создание и загрузка модели
    model = CharRNN(vocab_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = load_model(model, model_path, device)
    model.to(device)
    
    # Генерация текста
    while True:
        start_text = input("\nВведите начальный текст (или 'quit' для выхода): ").strip()
        if start_text.lower() == 'quit':
            break
        
        if not start_text:
            start_text = "машинное"
        
        try:
            temp = float(input("Температура (0.1-2.0, по умолчанию 1.0): ") or "1.0")
        except ValueError:
            temp = 1.0
            
        try:
            length = int(input("Длина текста (по умолчанию 200): ") or "200")
        except ValueError:
            length = 200
        
        generated = generate_text(model, char_to_idx, idx_to_char, 
                                start_str=start_text, length=length, 
                                temperature=temp, device=device)
        print(f"\nСгенерированный текст:\n{generated}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_from_cli()
    else:
        main()
