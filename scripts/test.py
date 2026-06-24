import os
import warnings
import subprocess
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers import logging

# === ГЛУШИТЕЛЬ ПРЕДУПРЕЖДЕНИЙ ===
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# --- 1. ПУТИ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_exe = os.path.join(current_dir, "ffmpeg", "bin", "ffmpeg.exe")

MODEL_DIR = "../models/aviation_whisper_v1_2"
AUDIO_FILE = "KLM900_1.wav"
TEMP_WAV = "temp_audio.wav"

# --- 2. ЖЕСТКАЯ КОНВЕРТАЦИЯ В WAV ---
print(f"1. Распаковка {AUDIO_FILE}...")
command = [
    ffmpeg_exe, "-y", "-i", AUDIO_FILE,
    "-ar", "16000", "-ac", "1", "-loglevel", "error", TEMP_WAV
]
subprocess.run(command, check=True)

# --- 3. ИНТЕЛЛЕКТУАЛЬНАЯ НАРЕЗКА ОТ ТИШИНЫ (VAD) ---
print("2. Поиск фраз (Silero VAD)...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Читаем звук прямо в память PyTorch
wav = read_audio(TEMP_WAV, sampling_rate=16000)
# Получаем список временных меток каждой реплики
speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)

print("3. Формирование отдельных предложений...")
audio_inputs = []
# Проходимся по каждой найденной фразе и вырезаем её
for ts in speech_timestamps:
    # Берем конкретный кусок звука от старта до конца фразы
    chunk_tensor = wav[ts['start']:ts['end']]
    # Превращаем в массив, который понимает конвейер
    chunk_numpy = chunk_tensor.numpy()
    audio_inputs.append({"raw": chunk_numpy, "sampling_rate": 16000})

# --- 4. ЗАГРУЗКА ВАШЕЙ МОДЕЛИ ---
print(f"4. Загрузка авиационной модели...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="cuda:0",
    torch_dtype=torch.float16
)
model.config.forced_decoder_ids = None

# --- 5. КОНВЕЙЕР ---
print("5. Создание Турбо-конвейера...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device=model.device
)

# --- 6. РАСПОЗНАВАНИЕ ---
print("6. Распознавание текста (пакетная обработка + УЛУЧШЕННАЯ ЛОГИКА)...")
results = asr_pipeline(
    audio_inputs,
    batch_size=8,
    generate_kwargs={
        "language": "english",
        "task": "transcribe",
        "condition_on_prev_tokens": False,

        # Оставляем только лучевой поиск (он сделает 90% работы по исправлению текста)
        "num_beams": 5
    }
)

# --- 7. ВЫВОД РЕЗУЛЬТАТА ---
print("\n=== РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ ===")
for res in results:
    text = res["text"].strip()
    if text:
        print(text)
print("===============================\n")

# Убираем за собой временный файл
if os.path.exists(TEMP_WAV):
    os.remove(TEMP_WAV)