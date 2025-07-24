import pandas as pd
from datasets import load_dataset

def parse_goemotions():
    """Загружает датасет GoEmotions и сохраняет в CSV"""
    print("Загрузка GoEmotions...")
    
    # Загружаем датасет
    dataset = load_dataset("go_emotions", "simplified")
    
    # Преобразуем в DataFrame
    train_data = dataset["train"]
    df = pd.DataFrame({
        "text": train_data["text"],
        "labels": train_data["labels"]
    })
    
    # Загружаем названия эмоций
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    
    # Преобразуем метки в названия
    df["emotion"] = df["labels"].apply(lambda x: emotion_labels[x[0]] if len(x) > 0 else "neutral")
    
    # Сохраняем в CSV
    df[["text", "emotion"]].to_csv("data/goemotions.csv", index=False)
    print("Датасет сохранён в data/goemotions.csv")

if __name__ == "__main__":
    parse_goemotions()
