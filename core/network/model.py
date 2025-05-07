# Основная модель ИИ
import torch
from transformers import AutoModelForCausalLM

class SinNetwork:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/base")
        self.context = []
        self.memory = []
        self.experience = 0
        self.level = 1

    def generate_response(self, prompt):
        full_context = "\n".join(self.context[-5:] + [prompt])
        inputs = self._prepare_inputs(full_context)
        outputs = self.model.generate(**inputs)
        return self._process_output(outputs)

    def learn_from_file(self, file_path):
        if file_path.endswith('.py'):
            self._learn_python_code(file_path)
        else:
            self._learn_text_data(file_path)

    def propose_self_update(self, new_code):
        from core.network.self_modifier import AdvancedCodeModifier
        modifier = AdvancedCodeModifier(self)
        return modifier.propose_change(new_code)
