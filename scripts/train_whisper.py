import os
import sys

# Кодеки ffmpeg
current_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_bin_path = os.path.join(current_dir, "ffmpeg", "bin")
os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ["PATH"]

import torch
import warnings
from datasets import load_dataset, Audio
from transformers import (
    BitsAndBytesConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

warnings.filterwarnings("ignore")

# --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
model_id = "openai/whisper-medium"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


if __name__ == "__main__":

    # ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
    print("Загрузка датасета...")
    dataset = load_dataset("jacktol/ATC-ASR-Dataset", split="train")

    # Ограничение датасета
    # dataset = dataset.select(range(150))

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print("Обработка аудиофайлов...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=2)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ЗАГРУЗКА МОДЕЛИ И ОПТИМИЗАЦИЯ ПАМЯТИ (LORA + 8-bit)
    print("Загрузка модели с квантованием...")

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ
    training_args = Seq2SeqTrainingArguments(
        output_dir="../backups/whisper-atc-model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_train_epochs=3,
        fp16=True,
        logging_steps=25,
        save_steps=100,
        remove_unused_columns=False,
        label_names=["labels"],
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # ЗАПУСК ОБУЧЕНИЯ
    print("СТАРТ ОБУЧЕНИЯ!...")
    model.config.use_cache = False
    trainer.train()

    # СОХРАНЕНИЕ
    model.save_pretrained("./aviation_whisper_V1")
    processor.save_pretrained("./aviation_whisper_V1")
    print("Обучение завершено - модель сохранена в папку ./aviation_whisper_V1")