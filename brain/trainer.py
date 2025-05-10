import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os

class SinTrainer:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        
    class ConversationDataset(Dataset):
        def __init__(self, texts, tokenizer, block_size=128):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt"
            )
            
        def __len__(self):
            return len(self.encodings["input_ids"])
            
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx]
            }
    
    def prepare_data(self, data_dir):
        texts = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    texts.extend(line.strip() for line in f if line.strip())
        return texts
    
    def train(self, data_dir, epochs=3, batch_size=4, lr=5e-5):
        texts = self.prepare_data(data_dir)
        if not texts:
            raise ValueError("No training data found")
            
        dataset = self.ConversationDataset(texts, self.model.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * epochs
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                
                outputs = self.model(inputs, attention_mask=masks)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1),
                    ignore_index=self.model.tokenizer.pad_token_id
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
        self.model.eval()
        return total_loss / len(dataloader)
