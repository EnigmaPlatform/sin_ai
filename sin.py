import os
import json
import torch
from pathlib import Path
from brain.model import SinModel
from brain.memory import SinMemory
from brain.trainer import SinTrainer

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
        self.load()

    def _load_model(self):
        model_path = self.models_dir / "sin_model.pt"
        if model_path.exists():
            return SinModel.load(model_path)
        return SinModel()

    def chat(self, user_input):
        self.memory.add_interaction(user_input, "")
        context = self.memory.get_context()
        prompt = f"{context}\nSin:"
        response = self.model.generate_response(prompt)
        self.memory.add_interaction(user_input, response)
        return response

    def train(self):
        dataset = self._load_all_datasets()
        if dataset is None:
            raise ValueError("No training data found")
        
        loss = self.trainer.train(dataset)
        self.save()
        return loss

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
                            for dialogue in data['dialogues']:
                                self.memory.add_dialogue(dialogue)
                elif filename.endswith('.txt'):
                    datasets.append(self.trainer.load_text_data(filepath))
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        return torch.utils.data.ConcatDataset(datasets) if datasets else None

    def save(self):
        self.model.save(self.models_dir / "sin_model.pt")
        self.memory.save(self.data_dir / "memory.json")

    def load(self):
        memory_path = self.data_dir / "memory.json"
        if memory_path.exists():
            self.memory.load(memory_path)
