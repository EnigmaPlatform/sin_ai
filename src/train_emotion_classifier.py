import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Загрузка данных
def load_data():
    df = pd.read_csv("data/goemotions.csv")
    # Убираем слишком короткие тексты
    df = df[df['text'].str.len() > 5]
    return df

# Подготовка данных для модели
def prepare_dataset(df):
    # Кодируем эмоции
    unique_emotions = df['emotion'].unique()
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    df['label'] = df['emotion'].map(emotion_to_id)
    
    # Разделение на train/test
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    return train_df, val_df, emotion_to_id, list(unique_emotions)

# Токенизация
def tokenize_data(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Метрики
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Основная функция обучения
def train_model():
    print("Загрузка данных...")
    df = load_data()
    train_df, val_df, emotion_to_id, emotion_names = prepare_dataset(df)
    
    print(f"Уникальные эмоции ({len(emotion_names)}): {emotion_names}")
    
    # Токенизатор и модель
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(emotion_names)
    )
    
    # Токенизация
    train_dataset = tokenize_data(train_df, tokenizer)
    val_dataset = tokenize_data(val_df, tokenizer)
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir="models/emotion_classifier",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Обучение
    print("Начало обучения...")
    trainer.train()
    
    # Сохранение модели и токенизатора
    trainer.save_model("models/emotion_classifier")
    tokenizer.save_pretrained("models/emotion_classifier")
    
    # Сохранение маппинга эмоций
    import json
    with open("models/emotion_classifier/emotion_map.json", "w") as f:
        json.dump({v: k for k, v in emotion_to_id.items()}, f)
    
    print("Модель обучена и сохранена!")

if __name__ == "__main__":
    train_model()
