import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingMonitor:
    def __init__(self, log_dir: Union[str, Path] = "data/training_logs"):
        """
        Комплексный мониторинг процесса обучения модели
        
        Args:
            log_dir: Директория для сохранения логов и графиков
        """
        self.logger = logging.getLogger(__name__)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log: Dict[str, Union[List, Dict]] = {}
        self.best_metrics: Dict[str, float] = {}
        self.start_time = datetime.now()
        
        try:
            self.log_dir.mkdir(exist_ok=True, parents=True)
            self.reset_logs()
            self._load_existing_logs()
            self.logger.info(f"Training monitor initialized at {self.log_dir}")
        except Exception as e:
            self.logger.critical(f"Failed to initialize monitor: {str(e)}", exc_info=True)
            raise

    def reset_logs(self) -> None:
        """Полностью сбрасывает логи тренировки"""
        self.current_log = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "metrics": {
                "accuracy": [],
                "f1": [],
                "precision": [],
                "recall": [],
            },
            "timestamps": []
        }
        self.best_metrics = {
            "train_loss": float('inf'),
            "val_loss": float('inf'),
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    def _load_existing_logs(self) -> None:
        """Загружает существующие логи из файла (если есть)"""
        log_file = self.log_dir / "training_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    self.current_log = data.get("current_log", {})
                    self.best_metrics = data.get("best_metrics", {})
                self.logger.info("Loaded existing training logs")
            except Exception as e:
                self.logger.warning(f"Failed to load existing logs: {str(e)}")
                self.reset_logs()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Логирует результаты эпохи
        
        Args:
            epoch: Номер эпохи
            train_loss: Значение функции потерь на обучении
            val_metrics: Метрики на валидации (словарь)
            learning_rate: Текущая скорость обучения
        """
        try:
            # Логируем основные данные
            self.current_log["epochs"].append(epoch)
            self.current_log["train_loss"].append(train_loss)
            self.current_log["timestamps"].append(datetime.now().isoformat())
            
            if learning_rate is not None:
                self.current_log["learning_rate"].append(learning_rate)
            
            # Обновляем лучшие показатели
            if train_loss < self.best_metrics["train_loss"]:
                self.best_metrics["train_loss"] = train_loss
            
            # Логируем валидационные метрики
            if val_metrics:
                for metric, value in val_metrics.items():
                    # Для val_loss особый случай
                    if metric == "val_loss":
                        self.current_log["val_loss"].append(value)
                        if value < self.best_metrics["val_loss"]:
                            self.best_metrics["val_loss"] = value
                    # Для остальных метрик
                    elif metric in self.current_log["metrics"]:
                        self.current_log["metrics"][metric].append(value)
                        if value > self.best_metrics.get(metric, -float('inf')):
                            self.best_metrics[metric] = value
            
            # Сохраняем и визуализируем
            self._save_log()
            self._plot_progress()
            self.logger.info(f"Logged epoch {epoch} | Train Loss: {train_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to log epoch {epoch}: {str(e)}", exc_info=True)
            raise

    def _save_log(self) -> None:
        """Сохраняет логи в JSON файл"""
        try:
            with open(self.log_dir / "training_log.json", 'w') as f:
                json.dump({
                    "current_log": self.current_log,
                    "best_metrics": self.best_metrics,
                    "training_time": str(datetime.now() - self.start_time),
                    "system_info": self._get_system_info()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save logs: {str(e)}", exc_info=True)
            raise

    def _plot_progress(self) -> None:
        """Генерирует и сохраняет графики прогресса обучения"""
        try:
            plt.figure(figsize=(18, 6))
            
            # График потерь
            plt.subplot(1, 3, 1)
            plt.plot(self.current_log["epochs"], self.current_log["train_loss"], 
                    'b-', label=f"Train (best: {self.best_metrics['train_loss']:.4f})")
            
            if self.current_log["val_loss"]:
                plt.plot(self.current_log["epochs"], self.current_log["val_loss"], 
                        'r-', label=f"Val (best: {self.best_metrics['val_loss']:.4f})")
            
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            # График метрик
            plt.subplot(1, 3, 2)
            colors = ['g', 'm', 'c', 'y']
            for i, (metric, values) in enumerate(self.current_log["metrics"].items()):
                if values:
                    best_val = self.best_metrics.get(metric, 0.0)
                    plt.plot(self.current_log["epochs"], values, 
                            f"{colors[i % len(colors)]}-",
                            label=f"{metric} (best: {best_val:.4f})")
            
            plt.title("Validation Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            
            # График скорости обучения (если есть данные)
            if self.current_log.get("learning_rate"):
                plt.subplot(1, 3, 3)
                plt.plot(self.current_log["epochs"], self.current_log["learning_rate"], 'k-')
                plt.title("Learning Rate Schedule")
                plt.xlabel("Epoch")
                plt.ylabel("Learning Rate")
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / "training_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {str(e)}", exc_info=True)
            raise

    def _get_system_info(self) -> Dict:
        """Возвращает информацию о системе"""
        import torch
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_cores": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            "timestamp": datetime.now().isoformat()
        }

    def get_best_epoch(self, metric: str = "accuracy") -> int:
        """
        Возвращает номер эпохи с лучшим значением метрики
        
        Args:
            metric: Название метрики (accuracy, f1 и т.д.)
            
        Returns:
            Номер лучшей эпохи (начиная с 1) или -1 если метрика не найдена
        """
        if not self.current_log["metrics"].get(metric):
            return -1
        return int(np.argmax(self.current_log["metrics"][metric])) + 1

    def get_best_metrics(self) -> Dict[str, float]:
        """Возвращает словарь с лучшими значениями всех метрик"""
        return self.best_metrics.copy()

    def get_last_metrics(self) -> Dict[str, float]:
        """Возвращает метрики последней эпохи"""
        last_metrics = {
            "train_loss": self.current_log["train_loss"][-1] if self.current_log["train_loss"] else None,
            "val_loss": self.current_log["val_loss"][-1] if self.current_log["val_loss"] else None,
        }
        
        for metric, values in self.current_log["metrics"].items():
            last_metrics[metric] = values[-1] if values else None
            
        return last_metrics

    def generate_report(self) -> Dict:
        """Генерирует полный отчет о тренировке"""
        return {
            "best_metrics": self.best_metrics,
            "last_metrics": self.get_last_metrics(),
            "training_duration": str(datetime.now() - self.start_time),
            "epochs_completed": len(self.current_log["epochs"]),
            "system_info": self._get_system_info(),
            "best_epoch": {
                "accuracy": self.get_best_epoch("accuracy"),
                "f1": self.get_best_epoch("f1")
            }
        }

    def save_report(self) -> Path:
        """Сохраняет отчет в файл и возвращает путь к нему"""
        report_path = self.log_dir / "training_report.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(self.generate_report(), f, indent=2)
            self.logger.info(f"Training report saved to {report_path}")
            return report_path
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}", exc_info=True)
            raise
