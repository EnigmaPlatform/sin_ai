import optuna
import torch
import logging
from typing import Dict, Any
from copy import deepcopy
from datetime import datetime
from pathlib import Path

class HyperparameterTuner:
    def __init__(self, model, train_loader, val_loader):
        self.base_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logging.getLogger("HyperparameterTuner")
        self.study_dir = Path("studies")
        self.study_dir.mkdir(exist_ok=True)

    def tune(self, n_trials=100, timeout=3600):
        """Основной метод оптимизации гиперпараметров"""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True
        )
        
        self._save_study(study)
        return study.best_params

    def _objective(self, trial):
        """Целевая функция для оптимизации"""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            "num_layers": trial.suggest_int("num_layers", 1, 8),
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        }
        
        model = deepcopy(self.base_model)
        model.apply_hyperparameters(params)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )
        
        for epoch in range(5):  # Короткие эпохи для быстрой оценки
            train_loss = self._train_epoch(model, optimizer)
            val_accuracy = self._validate(model)
            
            trial.report(val_accuracy, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return val_accuracy

    def _train_epoch(self, model, optimizer):
        model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            optimizer.zero_grad()
            outputs = model(batch["input"])
            loss = model.loss_function(outputs, batch["target"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate(self, model):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = model(batch["input"])
                _, predicted = torch.max(outputs.data, 1)
                total += batch["target"].size(0)
                correct += (predicted == batch["target"]).sum().item()
                
        return correct / total

    def _save_study(self, study):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_path = self.study_dir / f"study_{timestamp}.pkl"
        
        import joblib
        joblib.dump(study, study_path)
        self.logger.info(f"Saved study to {study_path}")
        
        # Визуализация
        self._visualize_study(study, study_path.with_suffix('.html'))

    def _visualize_study(self, study, output_path):
        import optuna.visualization as vis
        
        fig = vis.plot_optimization_history(study)
        fig.write_html(output_path)
