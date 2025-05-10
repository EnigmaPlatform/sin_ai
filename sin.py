import os
import json
import torch
from pathlib import Path
from brain.model import SinModel
from brain.memory import SinMemory
from brain.trainer import SinTrainer
from brain.evaluator import ModelEvaluator
from brain.monitor import TrainingMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR

class Sin:
    def __init__(self):
        self.data_dir = Path("data")
        self.models_dir = self.data_dir / "models"
        self.conversations_dir = self.data_dir / "conversations"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        
        self.model = self._load_model()
        self.memory = SinMemory()
        self.trainer = SinTrainer(self.model)
        self.evaluator = ModelEvaluator(self.model, self.model.tokenizer)
        self.monitor = TrainingMonitor()
        self.load()
        
        self.logger = logging.getLogger(__name__)
        try:
            self.data_dir = Path("data")
            self.models_dir = self.data_dir / "models"
            self.conversations_dir = self.data_dir / "conversations"
            
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.conversations_dir.mkdir(exist_ok=True)
            
            self.logger.info("Initializing model...")
            self.model = self._load_model()
            
            self.logger.info("Initializing memory...")
            self.memory = SinMemory()
            
            self.logger.info("Initializing trainer...")
            self.trainer = SinTrainer(self.model)
            
            self.logger.info("Initializing evaluator...")
            self.evaluator = ModelEvaluator(self.model, self.model.tokenizer)
            
            self.logger.info("Initializing monitor...")
            self.monitor = TrainingMonitor()
            
            self.logger.info("Loading saved state...")
            self.load()
            
            self.logger.info("Sin initialization complete")
            
        except Exception as e:
            self.logger.critical(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def evaluate(self, dataset, sample_size=100):
        """Оценка модели на датасете"""
        if not dataset:
            return {}
        return self.evaluator.evaluate_dataset(dataset, sample_size)

    def _load_model(self):
        model_path = self.models_dir / "sin_model.pt"
        if model_path.exists():
            return SinModel.load(model_path)
        return SinModel()

    def chat(self, user_input):
        """Улучшенная версия с очисткой контекста"""
        self.logger.info(f"Received user input: {user_input}")
    
        try:
        # Логируем добавление в память
            self.logger.debug("Adding interaction to memory")
            self.memory.add_interaction(user_input, "")
        
        # Формируем контекст
            context = "\n".join(list(self.memory.context)[-4:])
            self.logger.debug(f"Current context: {context}")
        
            prompt = f"{context}\nSin:"
            self.logger.debug(f"Generated prompt: {prompt}")
        
        # Генерация ответа
            self.logger.info("Generating response...")
            response = self.model.generate_response(prompt)
            self.logger.debug(f"Raw response: {response}")
        
        # Очистка ответа
            clean_response = response.split("Sin:")[-1].strip()
            clean_response = clean_response.split("\n")[0].strip()
            self.logger.debug(f"Cleaned response: {clean_response}")
        
        # Сохранение в память
            self.memory.add_interaction(user_input, clean_response)
            self.logger.info(f"Returning response: {clean_response}")
        
            return clean_response if clean_response else "Не могу сформулировать ответ"
        
        except Exception as e:
            self.logger.error(f"Error in chat(): {str(e)}", exc_info=True)
            return "Произошла ошибка при генерации ответа"

    def train(self, epochs=3, val_dataset=None):
        """Обучение с валидацией"""
        self.logger.info(f"Starting training for {epochs} epochs")
    
        try:
            train_dataset = self._load_all_datasets()
            if not train_dataset:
                error_msg = "No training data found"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
            self.logger.info(f"Loaded dataset with {len(train_dataset)} samples")
        
        # Валидация перед обучением
            if val_dataset:
                self.logger.info("Running initial validation...")
                init_metrics = self.evaluate(val_dataset)
                self.logger.info(f"Initial metrics: {init_metrics}")
        
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            scheduler = CosineAnnealingLR(optimizer, epochs)
        
            for epoch in range(epochs):
                self.logger.info(f"Starting epoch {epoch+1}/{epochs}")
                self.model.train()
                total_loss = 0
            
                try:
                    for batch_idx, batch in enumerate(self.trainer.get_data_loader(train_dataset)):
                        optimizer.zero_grad()
                        loss = self.trainer.train_step(batch)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                        if batch_idx % 50 == 0:
                            self.logger.debug(
                                f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}"
                        )
                
                    scheduler.step()
                
                # Валидация после эпохи
                    val_metrics = None
                    if val_dataset:
                        self.logger.info("Running validation...")
                        val_metrics = self.evaluate(val_dataset)
                        self.logger.info(f"Validation metrics: {val_metrics}")
                
                    self.monitor.log_epoch(epoch+1, total_loss, val_metrics)
                    self.logger.info(
                        f"Epoch {epoch+1} complete | Avg Loss: {total_loss/len(train_dataset):.4f}"
                )
                
                except Exception as e:
                    self.logger.error(f"Error during epoch {epoch+1}: {str(e)}", exc_info=True)
                    raise
        
            best_epoch = self.monitor.get_best_epoch()
            self.logger.info(f"Training complete! Best epoch: {best_epoch}")
            self.save()
            return self.monitor.current_log
    
        except Exception as e:
            self.logger.critical(f"Training failed: {str(e)}", exc_info=True)
            raise

    def _load_all_datasets(self):
        datasets = []
        for filename in os.listdir(self.conversations_dir):
            filepath = os.path.join(self.conversations_dir, filename)
            try:
                if filename.endswith('.json'):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'dialogues' in data:
                            datasets.append(self.trainer.load_json_data(filepath))
                elif filename.endswith('.txt'):
                    datasets.append(self.trainer.load_text_data(filepath))
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        return torch.utils.data.ConcatDataset(datasets) if datasets else None

    def save(self):
        """Сохранение модели и памяти"""
        self.model.save(self.models_dir / "sin_model.pt")
        self.memory.save(self.data_dir / "memory.json")

    def load(self):
        """Загрузка сохраненного состояния"""
        memory_path = self.data_dir / "memory.json"
        if memory_path.exists():
            self.memory.load(memory_path)

    def get_training_report(self):
        """Получение отчета о последнем обучении"""
        report_path = Path("data/logs/training_log.json")
        if report_path.exists():
            with open(report_path, "r") as f:
                return json.load(f)
        return None

    def compare_models(self, model_paths, test_dataset):
        """Сравнение нескольких версий моделей"""
        results = {}
        original_state = self.model.state_dict()
        
        try:
            for path in model_paths:
                self.model.load_state_dict(torch.load(path))
                metrics = self.evaluate(test_dataset)
                results[Path(path).name] = metrics
            
            self.model.load_state_dict(original_state)
            
            if len(results) > 1:
                base = next(iter(results.values()))
                for name, metrics in results.items():
                    results[name]["improvement"] = {
                        k: v - base[k] for k, v in metrics.items()
                    }
            
            return results
        except Exception as e:
            self.model.load_state_dict(original_state)
            raise e
