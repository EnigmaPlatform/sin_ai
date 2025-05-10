import torch
import os

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, name: str):
        torch.save(model.state_dict(), f"{self.model_dir}/{name}.pt")
    
    def load_model(self, name: str, network_class):
        model = network_class()
        model.model.load_state_dict(torch.load(f"{self.model_dir}/{name}.pt", map_location=model.device))
        return model
