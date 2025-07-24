import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
import random

# Загрузка модели эмоционального анализа
emotion_tokenizer = AutoTokenizer.from_pretrained("models/emotion_classifier")
emotion_model = AutoModelForSequenceClassification.from_pretrained("models/emotion_classifier")

# Загрузка генератора ответов
generator_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
generator_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Загрузка маппинга эмоций
with open("models/emotion_classifier/emotion_map.json", "r") as f:
    emotion_map = json.load(f)

# Функция классификации эмоций
def classify_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_id = torch.argmax(probs, dim=1).item()
        emotion = emotion_map[str(predicted_id)]
        confidence = probs[0][predicted_id].item()
    return emotion, confidence

# Функция генерации ответа
def generate_response(context, emotion):
    # Добавляем эмоциональный тег к контексту
    tagged_context = f"[{emotion}] {context}"
    
    inputs = generator_tokenizer.encode(tagged_context + generator_tokenizer.eos_token, return_tensors='pt')
    
    # Генерация с учётом длины и температуры
    reply_ids = generator_model.generate(
        inputs, 
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator_tokenizer.eos_token_id
    )
    
    response = generator_tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

# Главная функция чатбота
def chatbot(text):
    if not text.strip():
        return "Пожалуйста, введите текст для общения."
    
    # Определяем эмоцию
    emotion, confidence = classify_emotion(text)
    
    # Генерируем ответ
    response = generate_response(text, emotion)
    
    return f"Эмоция: {emotion} (уверенность: {confidence:.2f})\nОтвет: {response}"

# Создание интерфейса Gradio
interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Введите ваше сообщение...", label="Ваш текст"),
    outputs=gr.Textbox(label="Ответ бота"),
    title="🧠 Эмоциональный чат-бот",
    description="Бот, который понимает эмоции и отвечает с чувством!",
    examples=[
        ["Я так рад, что сегодня солнечно!"],
        ["Мне грустно, всё идёт не так..."],
        ["Ты такой успешный, а я нет..."],
        ["Я чувствую вдохновение!"]
    ]
)

if __name__ == "__main__":
    interface.launch(share=True)
