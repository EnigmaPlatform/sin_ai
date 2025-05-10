import numpy as np
from sin import Sin

def main():
    # Инициализация
    ai = Sin()
    
    # Обучение с валидацией
    train_data = [...]  # Загрузите тренировочные данные
    val_data = [...]    # Загрузите валидационные данные
    
    history = ai.train(epochs=5, val_dataset=val_data)
    
    # Анализ результатов
    print("\nTraining Report:")
    print(f"Best Accuracy: {max(history['metrics']['accuracy']):.2%}")
    print(f"Final Loss: {history['train_loss'][-1]:.4f}")
    
    # Сравнение моделей
    print("\nModel Comparison:")
    comparison = ai.compare_models(
        ["data/models/sin_model_v1.pt", "data/models/sin_model_v2.pt"],
        val_data
    )
    
    for name, metrics in comparison.items():
        print(f"\n{name}:")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Improvement: {metrics.get('improvement', {}).get('accuracy', 0):+.2%}")

if __name__ == "__main__":
    main()
