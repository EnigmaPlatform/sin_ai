# ui/visualizer.py

import time
from datetime import datetime
from typing import List
import logging

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class TrainingVisualizer:
    def __init__(self) -> None:
        self.start_time: datetime = None
        self.last_update: datetime = None
        self.total_steps: int = 0
        self.progress_history: List[float] = []
        self.eta_history: List[float] = []
    
    def start_training(self, total_steps: int) -> None:
        """Начало обучения с визуализацией."""
        self.start_time = datetime.now()
        self.last_update = self.start_time
        self.total_steps = total_steps
        self.progress_history = []
        self.eta_history = []
        
        logger.info(f"Starting training of {total_steps} steps")
    
    def update_progress(self, current_step: int) -> None:
        """Обновление прогресса обучения."""
        now = datetime.now()
        progress = (current_step / self.total_steps) * 100
        
        # Расчет ETA
        elapsed = (now - self.start_time).total_seconds()
        time_per_step = elapsed / current_step if current_step > 0 else 0
        remaining_steps = self.total_steps - current_step
        eta = remaining_steps * time_per_step
        
        self.progress_history.append(progress)
        self.eta_history.append(eta)
        self.last_update = now
        
        # Вывод в консоль
        self._print_progress(current_step, progress, eta)
    
    def complete_training(self) -> None:
        """Завершение обучения и сохранение графиков."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Training completed in {total_time:.2f} seconds")
        self._plot_progress()
    
    def _print_progress(self, current_step: int, progress: float, eta: float) -> None:
        """Вывод прогресса в консоль."""
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta > 0 else "--:--:--"
        logger.info(
            f"Progress: {progress:.1f}% | "
            f"Step: {current_step}/{self.total_steps} | "
            f"ETA: {eta_str}"
        )
    
    def _plot_progress(self) -> None:
        """Визуализация прогресса обучения."""
        if not self.progress_history:
            return
        
        plt.figure(figsize=(10, 5))
        
        # График прогресса
        plt.subplot(1, 2, 1)
        plt.plot(self.progress_history)
        plt.title("Training Progress")
        plt.xlabel("Step")
        plt.ylabel("Progress (%)")
        plt.grid(True)
        
        # График ETA
        plt.subplot(1, 2, 2)
        plt.plot(self.eta_history)
        plt.title("Estimated Time Remaining")
        plt.xlabel("Step")
        plt.ylabel("Seconds")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("data/training_progress.png")
        plt.close()
