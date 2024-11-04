# -*- coding: windows-1251 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Загрузка токенизатора и обученной модели
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")

# Включаем модель в режим оценки
model.cuda()

def generate_response(question):
    # Форматируем входной текст
    test = tokenizer.encode(question, return_tensors="pt")

    # Генерация ответа
    with torch.no_grad():
        # Вычисляем выходные данные
        output = model.generate(
            test.cuda(),
            repetition_penalty=6.0,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=1,
            no_repeat_ngram_size=2
        )

    # Декодируем и возвращаем ответ
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Ответ: ")[-1].strip()  # Извлекаем текст ответа

# Пример использования
while True:
    question = input("Введите вопрос (или 'exit' для выхода): ")
    if question.lower() == 'exit':
        break
    answer = generate_response(question)
    print(f"Ответ: {answer}")
