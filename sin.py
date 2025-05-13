import os
import json
import torch
import logging
import tempfile
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from logging.handlers import RotatingFileHandler
import transformers
from brain.model import SinModel
from brain.memory import SinMemory
from brain.trainer import SinTrainer, DialogDataset
from brain.evaluator import ModelEvaluator
from brain.monitor import TrainingMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from typing import Dict, Optional, List  # Добавьте это в существующие импорты

# Настройка кодировки для Windows
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

def setup_logging():
    """Настройка системы логирования с UTF-8"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Логи в файл с ротацией (с UTF-8)
    os.makedirs('data/logs', exist_ok=True)
    file_handler = RotatingFileHandler(
        'data/logs/sin.log',
        encoding='utf-8',
        maxBytes=5*1024*1024,
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    
    # Логи в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class Sin:
    def __init__(self, model_path=None):
        """
        Инициализация Sin AI
        
        Args:
            model_path (str, optional): Путь к конкретной модели для загрузки. 
                                      Если None, загрузит последнюю доступную модель.
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.data_dir = Path("data")
            self.models_dir = self.data_dir / "models"
            self.conversations_dir = self.data_dir / "conversations"
            self.logs_dir = self.data_dir / "logs"
            
            # Создаем необходимые директории
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.conversations_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            
            self.logger.info("Initializing model...")
            self.model = self._load_model(model_path)
            
            # Инициализация специальных токенов
            self._initialize_special_tokens()
            
            self.logger.info("Initializing memory...")
            self.memory = SinMemory()
            
            self.logger.info("Initializing trainer...")
            self.trainer = SinTrainer(self.model)
            
            self.logger.info("Initializing evaluator...")
            self.evaluator = ModelEvaluator(self.model, self.model.tokenizer)
            
            self.logger.info("Initializing monitor...")
            self.monitor = TrainingMonitor(log_dir="data/training_logs")
            
            self.logger.info("Loading saved state...")
            self.load()
            
            self.logger.info("Sin initialization complete")
            
        except Exception as e:
            self.logger.critical(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def _initialize_special_tokens(self):
        """Инициализация специальных токенов, если их нет"""
        if hasattr(self.model, 'tokenizer'):
            if not hasattr(self.model.tokenizer, 'sep_token') or not self.model.tokenizer.sep_token:
                self.model.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
                self.logger.info("Added missing SEP token")
            if not hasattr(self.model.tokenizer, 'cls_token') or not self.model.tokenizer.cls_token:
                self.model.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
                self.logger.info("Added missing CLS token")

    def _load_model(self, model_path=None):
        """
        Загружает модель из указанного пути или находит последнюю версию
        
        Args:
            model_path (str, optional): Путь к конкретной модели. Если None, ищет последнюю.
            
        Returns:
            SinModel: Загруженная или новая модель
        """
        if model_path:
            model_path = Path(model_path)
            if model_path.exists():
                if model_path.stat().st_size < 1024:  # Если файл слишком маленький
                    model_path.unlink()  # Удаляем поврежденный файл
                    self.logger.warning(f"Removed corrupted model: {model_path}")
                    return SinModel()
                return self._load_single_model(model_path)
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Автоматический поиск последней модели
        model_files = list(self.models_dir.glob('*.pt'))
        if not model_files:
            self.logger.info("No models found, creating new one")
            return SinModel()
            
        # Сортировка по timestamp в имени файла
        def extract_timestamp(path):
            try:
                name = path.stem
                if '_' in name:
                    ts_part = name.split('_')[-1]
                    return datetime.strptime(ts_part, "%Y%m%d%H%M%S")
                return datetime.fromtimestamp(path.stat().st_mtime)  # Используем время модификации файла как fallback
            except:
                return datetime.fromtimestamp(0)  # Всегда возвращаем datetime
                
        model_files.sort(key=lambda x: extract_timestamp(x), reverse=True)
        
        # Попробуем загрузить модели по порядку
        for model_file in model_files:
            try:
                model = self._load_single_model(model_file)
                self.logger.info(f"Successfully loaded model: {model_file.name}")
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load {model_file.name}: {str(e)}")
                continue
                
        self.logger.info("All model loading attempts failed, creating new model")
        return SinModel()

    def _load_single_model(self, model_path):
        """Загружает одну модель с проверкой целостности"""
        self.logger.info(f"Loading model from {model_path}")
        
        # Проверка размера файла
        file_size = model_path.stat().st_size
        if file_size < 1024:
            raise ValueError(f"Model file too small ({file_size} bytes), likely corrupted")
            
        # Загрузка данных
        data = torch.load(model_path, map_location='cpu')
        
        # Проверка структуры данных
        required_keys = {'model_state', 'tokenizer_config'}
        if not all(k in data for k in required_keys):
            raise ValueError("Model file missing required data")
            
        # Создание и загрузка модели
        model = SinModel()
        model.load_state_dict(data['model_state'])
        
        # Загрузка токенизатора
        if hasattr(model, 'tokenizer'):
            special_tokens = data['tokenizer_config'].get('special_tokens', {})
            if special_tokens:
                model.tokenizer.add_special_tokens(special_tokens)
                self.logger.info(f"Added {len(special_tokens)} special tokens")
            
        return model

    def chat(self, user_input):
        """Улучшенная версия чата с обработкой команд и ошибок"""
        try:
            # Обработка команд
            if user_input.startswith('/'):
                return self._handle_command(user_input)

            if not user_input or not user_input.strip():
                return "Пожалуйста, введите осмысленный запрос"
                
            self.logger.info(f"Received user input: {user_input}")
            
            # Логируем добавление в память
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
            
            # Логирование с обработкой Unicode
            try:
                self.logger.debug(f"Cleaned response: {clean_response}")
                self.logger.info(f"Returning response: {clean_response}")
            except UnicodeEncodeError:
                self.logger.info("Returning response (unicode characters omitted)")
        
            # Сохранение в память
            self.memory.add_interaction(user_input, clean_response)
            
            return clean_response if clean_response else "Не могу сформулировать ответ"
        
        except Exception as e:
            self.logger.error(f"Error in chat(): {str(e)}", exc_info=True)
            return "Произошла ошибка при генерации ответа"

    def _handle_command(self, command):
        """Обработка всех команд чата"""
        parts = command.split()
        if not parts:
            return "Неизвестная команда"
            
        cmd = parts[0].lower()
        
        if cmd == "/train":
            return self._handle_train_command()
        elif cmd == "/save":
            model_name = parts[1] if len(parts) > 1 else None
            save_path = self.save_model(model_name)
            return f"Модель сохранена как: {save_path}"
        elif cmd == "/models":
            models = self.list_models()
            return "\nДоступные модели:\n" + "\n".join(f"{i}. {m}" for i, m in enumerate(models, 1))
        elif cmd == "/model_info":
            models_info = self.get_model_info()
            report = []
            for info in models_info:
                model_info = f"{info.get('name', 'N/A')}\n"
                model_info += f"  Размер: {info.get('size', 0) / 1024 / 1024:.2f} MB\n"
                modified = info.get('modified', datetime.now())
                model_info += f"  Изменена: {modified.strftime('%Y-%m-%d %H:%M:%S') if isinstance(modified, datetime) else modified}\n"
                report.append(model_info)
            return "\nИнформация о моделях:\n" + "\n".join(report)
        elif cmd == "/load" and len(parts) > 1:
            model_name = parts[1]
            self.model = self._load_model(self.models_dir / model_name)
            return f"Модель {model_name} загружена"
        elif cmd == "/reset":
            self.memory.context.clear()
            return "История диалога очищена"
        elif cmd == "/memory":
            return "\nТекущая память:\n" + "\n".join(f"{i}. {msg}" for i, msg in enumerate(self.memory.context, 1))
        elif cmd == "/report":
            report = self.get_training_report()
            if report:
                return "\nОтчет о последнем обучении:\n" + json.dumps(report, indent=2, ensure_ascii=False)
            return "Отчет об обучении не найден"
        elif cmd == "/config":
            config_info = [
                f"Размер словаря: {len(self.model.tokenizer)}",
                f"Параметры модели: {sum(p.numel() for p in self.model.parameters())}",
                f"CUDA доступно: {torch.cuda.is_available()}",
                f"Устройство модели: {next(self.model.parameters()).device}"
            ]
            return "\nКонфигурация модели:\n" + "\n".join(f"  {info}" for info in config_info)
        elif cmd == "/help":
            help_text = [
                "\nДоступные команды:",
                "  /help - показать это сообщение",
                "  /save [имя] - сохранить модель (с опциональным именем)",
                "  /models - список доступных моделей",
                "  /model_info - подробная информация о моделях",
                "  /load <имя> - загрузить другую модель",
                "  /reset - очистить историю диалога",
                "  /train - начать обучение",
                "  /memory - показать текущую память",
                "  /report - показать отчет о последнем обучении",
                "  /config - показать конфигурацию модели",
                "  /exit - выйти из программы"
            ]
            return "\n".join(help_text)
        else:
            return "Неизвестная команда. Напишите /help для списка команд"

    def _handle_train_command(self):
        """Обработка команды обучения из чата"""
        try:
            self.logger.info("Starting training from chat command")
            train_log = self.train(epochs=3)
            best_epoch = self.monitor.get_best_epoch("accuracy")
            best_metrics = self.monitor.get_best_metrics()
            
            return (f"Обучение завершено!\n"
                   f"Лучшая эпоха: {best_epoch}\n"
                   f"Точность: {best_metrics.get('accuracy', 0):.4f}\n"
                   f"Потери: {best_metrics.get('val_loss', 0):.4f}")
                   
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            return "Ошибка при обучении модели. Проверьте логи для деталей."

    def train(self, epochs=3, val_dataset=None):
        """Обучение с валидацией"""
         # === Оптимизация для CPU ===
        import os
        import psutil
        torch.set_num_threads(min(4, os.cpu_count() or 1))  # Ограничение потоков
        torch.backends.quantized.engine = 'qnnpack'         # Для мобильных CPU
        os.environ['OMP_NUM_THREADS'] = '1'                 # Для OpenMP
        os.environ['MKL_NUM_THREADS'] = '1'                 # Для Intel MKL
    
        self.logger.info(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")

        try:
            train_dataset = self._load_all_datasets()
            if not train_dataset:
                error_msg = "No training data found"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"Loaded dataset with {len(train_dataset)} samples")
        
        # Создаем DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=2,              # Маленький batch для слабого CPU
                shuffle=True,
                num_workers=0,             # 0 для избежания ошибок на ноутбуках
                pin_memory=False,          # Не использовать для CPU
                collate_fn=self.trainer._collate_fn
)
        
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            scheduler = CosineAnnealingLR(optimizer, epochs)
        
            for epoch in range(epochs):
    self.model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        loss = self.trainer.train_step(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Логирование каждые 10 батчей
        if batch_idx % 10 == 0:
            self.logger.info(
                f"Batch {batch_idx} | Loss: {loss.item():.4f} | "
                f"RAM: {psutil.virtual_memory().percent}%"
            )
        
                # Валидация после эпохи
                    val_metrics = None
                    if val_dataset:
                        self.logger.info("Running validation...")
                        val_metrics = self.trainer.evaluate(val_dataset)
                        self.logger.info(f"Validation metrics: {val_metrics}")
        
                # Логирование прогресса
                    self.monitor.log_epoch(
                        epoch=epoch+1,
                        train_loss=total_loss/len(train_loader),
                        val_metrics=val_metrics,
                        learning_rate=scheduler.get_last_lr()[0]
                )
        
                    self.logger.info(
                        f"Epoch {epoch+1} complete | Avg Loss: {total_loss/len(train_loader):.4f}"
                )
        
                except Exception as e:
                    self.logger.error(f"Error during epoch {epoch+1}: {str(e)}", exc_info=True)
                    raise
    
        # После завершения обучения
            best_epoch = self.monitor.get_best_epoch("accuracy")
            best_metrics = self.monitor.get_best_metrics()
    
            self.logger.info(f"Training complete! Best epoch: {best_epoch}")
            self.logger.info(f"Best metrics: {best_metrics}")
    
        # Сохранение модели и отчетов
            self.save()
            self.monitor.save_report()
        
            return best_metrics

        except Exception as e:
            self.logger.critical(f"Training failed: {str(e)}", exc_info=True)
            raise

    def _load_all_datasets(self):
        """Загружает все доступные датасеты для обучения"""
        datasets = []
    
    # Проверяем существование папки
        if not self.conversations_dir.exists():
            self.logger.error(f"Directory not found: {self.conversations_dir}")
            return None

        for file in self.conversations_dir.glob('*'):
            try:
                if file.suffix.lower() != '.json':
                    self.logger.warning(f"Skipping non-JSON file: {file.name}")
                    continue
                
                self.logger.info(f"Processing file: {file.name}")
            
            # Читаем содержимое файла
                with open(file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                # Проверяем структуру данных
                    if isinstance(content, dict) and 'dialogues' in content:
                        dialogues = content['dialogues']
                        if not isinstance(dialogues, list):
                            self.logger.warning(f"Invalid 'dialogues' format in {file.name}")
                            continue
                    
                    # Создаем датасет
                        try:
                            dataset = DialogDataset(dialogues, self.model.tokenizer)
                            datasets.append(dataset)
                            self.logger.info(f"Successfully loaded {len(dialogues)} dialogues from {file.name}")
                        except Exception as e:
                            self.logger.error(f"Error creating dataset from {file.name}: {str(e)}")
                            continue
                    else:
                        self.logger.warning(f"Invalid JSON structure in {file.name} - missing 'dialogues' field")
                        continue
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in {file.name}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
                continue
    
        if datasets:
            combined_dataset = torch.utils.data.ConcatDataset(datasets)
            self.logger.info(f"Combined dataset contains {len(combined_dataset)} samples")
            return combined_dataset
    
        self.logger.error("No valid datasets found after processing all files")
        return None


    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Оценка модели на датасете"""
        self.model.eval()
        dataloader = self.trainer.get_data_loader(dataset, shuffle=False)
        total_loss = 0
    
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.model.device)
                masks = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
            
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                total_loss += outputs['loss'].item()
    
        return {'loss': total_loss / len(dataloader)}

    def save(self):
        """Автоматическое сохранение модели с timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = self.models_dir / f"model_{timestamp}.pt"
        
        # Подготовка данных для сохранения
        save_data = {
            'model_state': self.model.state_dict(),
            'tokenizer_config': {
                'vocab': getattr(self.model.tokenizer, 'get_vocab', lambda: {})(),
                'special_tokens': getattr(self.model.tokenizer, 'special_tokens_map', {}),
                'added_tokens': getattr(self.model.tokenizer, 'added_tokens_encoder', {})
            },
            'metadata': {
                'saved_at': timestamp,
                'version': '2.0',
                'training_stats': getattr(self.monitor, 'current_log', None),
                'system': {
                    'python': sys.version,
                    'torch': torch.__version__,
                    'transformers': transformers.__version__
                }
            }
        }
        
        # Атомарное сохранение через временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=self.models_dir, suffix='.tmp')
        try:
            torch.save(save_data, temp_file)
            temp_file.close()
            
            # Атомарная операция переименования
            os.replace(temp_file.name, model_path)
            self.logger.info(f"Model successfully saved to {model_path.name}")
            
            # Очистка старых моделей
            self._cleanup_old_models(max_models=5)
            
            # Сохраняем память
            self._save_memory()
            
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise

    def _save_memory(self):
        """Сохраняет память в атомарном режиме"""
        memory_path = self.data_dir / "memory.json"
        temp_path = memory_path.with_suffix('.tmp')
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "context": list(self.memory.context),
                    "long_term": self.memory.long_term,
                    "knowledge_graph": self.memory.knowledge_graph
                }, f, ensure_ascii=False, indent=2)
                
            os.replace(temp_path, memory_path)
            self.logger.info("Memory saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save memory: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load(self):
        """Загрузка сохраненного состояния"""
        try:
            memory_path = self.data_dir / "memory.json"
            if memory_path.exists():
                self.logger.info("Loading memory...")
                with open(memory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory.context = deque(data.get("context", []), maxlen=self.memory.context.maxlen)
                    self.memory.long_term = data.get("long_term", [])
                    self.memory.knowledge_graph = data.get("knowledge_graph", [])
                self.logger.info("Memory loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load memory: {str(e)}", exc_info=True)
            # Не прерываем работу, продолжаем с пустой памятью
            self.memory = SinMemory()

    def get_training_report(self):
        """Получение отчета о последнем обучении"""
        report_path = self.data_dir / "training_logs" / "training_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding='utf-8') as f:
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

    def save_model(self, model_name=None):
        """Ручное сохранение модели с указанием имени"""
        try:
            if not model_name:
                model_name = f"sin_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            elif not model_name.endswith('.pt'):
                model_name += '.pt'
            
            model_path = self.models_dir / model_name
            self.model.save(model_path)
            self.logger.info(f"Model manually saved to {model_path}")
            
            # Очистка старых моделей (сохраняем последние 5)
            self._cleanup_old_models(max_models=5)
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Manual save failed: {str(e)}", exc_info=True)
            raise

    def _cleanup_old_models(self, max_models=5):
        """Удаление старых моделей, кроме max_models последних"""
        try:
            model_files = list(self.models_dir.glob('*.pt'))
            if len(model_files) <= max_models:
                return
                
            # Сортировка по timestamp в имени
            def extract_timestamp(path):
                try:
                    name = path.stem
                    if '_' in name:
                        ts_part = name.split('_')[-1]
                        return datetime.strptime(ts_part, "%Y%m%d%H%M%S")
                    return datetime.fromtimestamp(path.stat().st_mtime)  # Используем время модификации файла как fallback
                except:
                    return datetime.fromtimestamp(0)  # Всегда возвращаем datetime
                    
            model_files.sort(key=lambda x: extract_timestamp(x))
            
            # Удаляем самые старые
            for old_model in model_files[:-max_models]:
                try:
                    old_model.unlink()
                    self.logger.info(f"Removed old model: {old_model.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {old_model.name}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error during model cleanup: {str(e)}")

    def list_models(self):
        """Список доступных моделей"""
        return [f.name for f in sorted(
            self.models_dir.glob('*.pt'),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )]

    def get_model_info(self):
        """Возвращает информацию о всех доступных моделях"""
        models = []
        for model_file in self.models_dir.glob('*.pt'):
            try:
                # Читаем только метаданные без загрузки всей модели
                data = torch.load(model_file, map_location='cpu', weights_only=True)
                models.append({
                    'path': str(model_file),
                    'name': model_file.name,
                    'size': model_file.stat().st_size,
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime),
                    'metadata': data.get('metadata', {})
                })
            except Exception as e:
                self.logger.warning(f"Could not read metadata from {model_file.name}: {str(e)}")
                models.append({
                    'path': str(model_file),
                    'name': model_file.name,
                    'size': model_file.stat().st_size,
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime),
                    'error': str(e)
                })
        
        # Сортировка по дате изменения (новые сначала)
        models.sort(key=lambda x: x['modified'], reverse=True)
        return models
