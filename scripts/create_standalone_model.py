from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

import os

# 1. Указываем локальные пути (относительно файла, который вы запускаете)

checkpoint_path = "../models/model"

standalone_path = "../models/Aviation_Whisper_V2_2"

# 2. Проверка, существует ли папка с чекпоинтом

if not os.path.exists(checkpoint_path):

    print(f"❌ Ошибка: Папка {checkpoint_path} не найдена. Проверьте правильность пути!")

    exit()

print("Загрузка базовой модели (может занять время, если скачивается впервые)...")

# Загружаем базовую модель на процессор (CPU)

base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", device_map="cpu")

print(f"Применение адаптера из {checkpoint_path}...")

# Накладываем ваш обученный чекпоинт

peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

print("Слияние весов (Merge & Unload)...")

# МАГИЯ: Сливаем веса в единую нейросеть

merged_model = peft_model.merge_and_unload()

print("Сохранение независимой офлайн-модели...")

# Сохраняем готовую модель

os.makedirs(standalone_path, exist_ok=True)

merged_model.save_pretrained(standalone_path)

# Сохраняем процессор (правила токенизации и обработки звука)

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

processor.save_pretrained(standalone_path)

print(f"✅ Успешно! Монолитная модель сохранена в: {standalone_path}")