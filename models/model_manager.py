import torch
import os

class ModelManager:
    def __init__(self, save_dir="models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model, name):
        torch.save(model.state_dict(), f"{self.save_dir}/{name}.pt")
    
    def load_model(self, name):
        model = SinNetwork()
        model.model.load_state_dict(torch.load(f"{self.save_dir}/{name}.pt"))
        return model
