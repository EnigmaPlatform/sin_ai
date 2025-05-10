import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm
from torch.nn.functional import cross_entropy

class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def calculate_accuracy(self, predictions, targets):
        """Вычисление точности предсказаний"""
        preds = np.argmax(predictions.cpu().numpy(), axis=1)
        return accuracy_score(targets.cpu().numpy(), preds)

    def evaluate_response_quality(self, generated, reference):
        """Оценка качества сгенерированного ответа (0-1)"""
        inputs_gen = self.tokenizer(generated, return_tensors='pt').to(self.device)
        inputs_ref = self.tokenizer(reference, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs_gen = self.model(**inputs_gen)
            outputs_ref = self.model(**inputs_ref)
            
            # Косинусная схожесть последних скрытых состояний
            sim = torch.cosine_similarity(
                outputs_gen.hidden_states[-1].mean(dim=1),
                outputs_ref.hidden_states[-1].mean(dim=1)
            ).item()
            
            # Perplexity оценки
            ppl = self._calculate_perplexity(outputs_gen.logits, inputs_gen.input_ids)
            
        return {
            'similarity': max(0, min(1, (sim + 1) / 2)),  # Нормализация к [0,1]
            'perplexity': ppl
        }

    def _calculate_perplexity(self, logits, labels):
        """Вычисление perplexity"""
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        return torch.exp(loss).item()

    def evaluate_dataset(self, dataset, sample_size=100):
        """Комплексная оценка на датасете"""
        results = {
            'loss': 0.0,
            'accuracy': 0.0,
            'semantic_similarity': 0.0,
            'count': 0
        }
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
                if i >= sample_size // 4:
                    break
                    
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                
                outputs = self.model(inputs, attention_mask=masks)
                
                # Вычисление метрик
                loss = cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                ).item()
                
                acc = self.calculate_accuracy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1)
                )
                
                results['loss'] += loss
                results['accuracy'] += acc
                results['count'] += 1
        
        if results['count'] > 0:
            results = {k: v / results['count'] for k, v in results.items()}
        return results
