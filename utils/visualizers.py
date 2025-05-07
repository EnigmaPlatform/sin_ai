# Визуализация прогресса
class TrainingVisualizer:
    @staticmethod
    def show_progress(epoch, total, metrics):
        progress = int(epoch / total * 50)
        print(f"[{'#' * progress}{'.' * (50 - progress)}]")
        print("Метрики обучения:")
        for name, value in metrics.items():
            print(f"{name}: {'▮' * int(value * 10)} {value:.2f}")
    
    @staticmethod
    def show_knowledge_map(skills):
        print("\nКарта знаний:")
        for skill, level in skills.items():
            print(f"{skill.ljust(15)}: {'★' * level}{'☆' * (5 - level)}")
