import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

class TrainingMonitor:
    def __init__(self, log_dir="data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.current_log = {
            "epochs": [],
            "loss": [],
            "metrics": {}
        }
    
    def log_epoch(self, epoch, loss, metrics=None):
        self.current_log["epochs"].append(epoch)
        self.current_log["loss"].append(loss)
        if metrics:
            for k, v in metrics.items():
                if k not in self.current_log["metrics"]:
                    self.current_log["metrics"][k] = []
                self.current_log["metrics"][k].append(v)
        
        self._save_log()
        self._plot_progress()
    
    def _save_log(self):
        with open(self.log_dir / "training_log.json", "w") as f:
            json.dump(self.current_log, f)
    
    def _plot_progress(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.current_log["epochs"], self.current_log["loss"], label="Loss")
        
        for metric, values in self.current_log["metrics"].items():
            plt.plot(self.current_log["epochs"], values, label=metric)
        
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.savefig(self.log_dir / "training_progress.png")
        plt.close()
