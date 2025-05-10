# examples/usage_examples.py
import numpy as np
from sin import Sin

def training_monitoring_example():
    # Инициализация и обучение
    ai = Sin()
    ai.train()  # После этого в data/logs появится отчет
    
    # Анализ результатов
    report = ai.get_training_report()
    if report:
        best_epoch = np.argmin(report['loss']) + 1
        print(f"\nАнализ обучения:")
        print(f"- Лучшая эпоха: {best_epoch}")
        print(f"- Минимальные потери: {min(report['loss']):.4f}")
        
        # Визуализация
        import matplotlib.pyplot as plt
        plt.plot(report['epochs'], report['loss'])
        plt.title('График обучения')
        plt.show()

def model_comparison_example():
    ai = Sin()
    
    # Создаем тестовые данные (можно заменить на реальные)
    test_data = [...]  # Список тестовых вопросов
    
    # Сравнение моделей
    print("\nСравнение моделей:")
    comparison = ai.compare_models("sin_model_v1.pt", 
                                 "sin_model_v2.pt",
                                 test_data)
    
    print(f"Точность v1: {comparison['sin_model_v1.pt']['accuracy']:.2%}")
    print(f"Точность v2: {comparison['sin_model_v2.pt']['accuracy']:.2%}")
    print(f"Разница: {comparison['difference']['accuracy']:+.2%}")

if __name__ == "__main__":
    training_monitoring_example()
    model_comparison_example()
