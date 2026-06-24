import os
import warnings
import subprocess
import threading
import sqlite3
import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
import pygame
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers import logging
import math
import time

# === НАСТРОЙКИ ===
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE = os.path.join(CURRENT_DIR, "ffmpeg", "bin", "ffmpeg.exe")
MODELS_BASE_DIR = "models"
DB_FILE = "database/transcriptions.db"

# Убедимся, что папки существуют
os.makedirs(MODELS_BASE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

# === ПАЛИТРА===
BG_COLOR = "#121212"
PANEL_COLOR = "#1e1e1e"
TEXTBOX_BG = "#181818"
ACCENT_COLOR = "#2fa572"
ACCENT_HOVER = "#248259"
SECONDARY_BTN = "#333333"
SECONDARY_HOVER = "#444444"
TEXT_COLOR = "#e0e0e0"


# === БАЗА ДАННЫХ ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, text TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


def save_to_db(filename, text):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (filename, text) VALUES (?, ?)", (filename, text))
    conn.commit()
    conn.close()


def get_history(search_query=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if search_query:
        c.execute(
            "SELECT id, filename, text, timestamp FROM history WHERE filename LIKE ? OR text LIKE ? ORDER BY timestamp DESC",
            (f"%{search_query}%", f"%{search_query}%"))
    else:
        c.execute("SELECT id, filename, text, timestamp FROM history ORDER BY timestamp DESC")
    data = c.fetchall()
    conn.close()
    return data


def delete_from_db(record_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


# === ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ===
class AviationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Aviation Transcriber Pro")
        self.geometry("950x650")
        self.resizable(False, False)
        self.configure(fg_color=BG_COLOR)
        ctk.set_appearance_mode("dark")

        # Инициализация аудиоплеера
        pygame.mixer.init()

        self.selected_file = None
        self.pipeline = None
        self.vad_model = None
        self.utils = None
        self.model_loaded = False
        self.is_playing = False

        self.setup_ui()
        init_db()

        # Первичная загрузка модели
        self.reload_model()

    def get_available_models(self):
        models = [d for d in os.listdir(MODELS_BASE_DIR) if os.path.isdir(os.path.join(MODELS_BASE_DIR, d))]
        if not models:
            models = ["openai/whisper-small"]
        return models

    def set_app_status(self, text, color="#888888"):
        self.after(0, lambda: self.app_status_label.configure(text=text, text_color=color))

    def set_model_status(self, text, color="#888888"):
        self.after(0, lambda: self.model_status_label.configure(text=text, text_color=color))

    def setup_ui(self):
        # --- ВЕРХНЯЯ ПАНЕЛЬ ---
        self.top_frame = ctk.CTkFrame(self, fg_color=PANEL_COLOR, corner_radius=12)
        self.top_frame.pack(side="top", fill="x", padx=20, pady=(20, 10))

        # Левый блок: Выбор модели
        self.model_frame = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        self.model_frame.pack(side="left", padx=15, pady=15)

        ctk.CTkLabel(self.model_frame, text="Модель:", text_color="#888", font=ctk.CTkFont(size=11)).pack(anchor="w")

        self.model_var = ctk.StringVar(value=self.get_available_models()[0])
        self.model_menu = ctk.CTkOptionMenu(self.model_frame, values=self.get_available_models(),
                                            variable=self.model_var,
                                            command=self.on_model_change, fg_color=SECONDARY_BTN, width=160)
        self.model_menu.pack()

        # Инструменты аудио
        self.btn_select = ctk.CTkButton(self.top_frame, text="📁 Аудио", command=self.select_file,
                                        fg_color=SECONDARY_BTN, hover_color=SECONDARY_HOVER, width=90, height=36)
        self.btn_select.pack(side="left", padx=(10, 5))

        self.btn_play = ctk.CTkButton(self.top_frame, text="🎧 Слушать", command=self.toggle_audio, state="disabled",
                                      fg_color="#17a2b8", hover_color="#138496", width=90, height=36)
        self.btn_play.pack(side="left", padx=5)

        self.btn_start = ctk.CTkButton(self.top_frame, text="▶️ Старт", command=self.start_transcription,
                                       state="disabled",
                                       fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER, width=90, height=36)
        self.btn_start.pack(side="left", padx=5)

        self.file_label = ctk.CTkLabel(self.top_frame, text="Файл не выбран", text_color="#666", width=140, anchor="w")
        self.file_label.pack(side="left", padx=10)

        # Правый блок: История и Копирование
        self.btn_db = ctk.CTkButton(self.top_frame, text="🗄️ История", command=self.show_database,
                                    fg_color=SECONDARY_BTN, hover_color=SECONDARY_HOVER, width=80, height=36)
        self.btn_db.pack(side="right", padx=15)

        self.btn_copy = ctk.CTkButton(self.top_frame, text="📋 Копировать", command=self.copy_to_clipboard,
                                      fg_color="transparent", border_width=1, border_color=SECONDARY_HOVER,
                                      hover_color=SECONDARY_BTN, width=110, height=36)
        self.btn_copy.pack(side="right", padx=(5, 0))

        # --- ЦЕНТРАЛЬНАЯ ПАНЕЛЬ (Поле для текста) ---
        self.text_frame = ctk.CTkFrame(self, fg_color=PANEL_COLOR, corner_radius=12)
        self.text_frame.pack(fill="both", expand=True, padx=20, pady=(5, 5))

        self.textbox = ctk.CTkTextbox(self.text_frame, font=ctk.CTkFont(family="Consolas", size=15),
                                      wrap="word", fg_color=TEXTBOX_BG, text_color=TEXT_COLOR,
                                      corner_radius=8, border_spacing=10)
        self.textbox.pack(fill="both", expand=True, padx=2, pady=2)
        self.textbox.insert("0.0", "Ожидание аудиоданных...\n")

        # Настройка тегов для пословного выделения
        self.textbox.tag_config("timestamp", foreground=ACCENT_COLOR)
        self.textbox.tag_config("conf_high", foreground=TEXT_COLOR)
        self.textbox.tag_config("conf_mid", foreground="#ffc107")
        self.textbox.tag_config("conf_low", foreground="#dc3545")

        # --- НИЖНЯЯ ПАНЕЛЬ (Индикаторы) ---
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(side="bottom", fill="x", padx=25, pady=(5, 15))

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, height=3, progress_color=ACCENT_COLOR,
                                               fg_color="#333", mode="indeterminate")
        self.progress_bar.pack(side="top", fill="x", pady=(0, 8))
        self.progress_bar.set(0)

        self.app_status_label = ctk.CTkLabel(self.status_frame, text="Готов к работе", text_color="#666",
                                             font=ctk.CTkFont(size=12))
        self.app_status_label.pack(side="left")

        self.model_status_label = ctk.CTkLabel(self.status_frame, text="⏳ Ожидание ядра...", text_color="#e5a50a",
                                               font=ctk.CTkFont(size=12, weight="bold"))
        self.model_status_label.pack(side="right")

    # --- АУДИО ПЛЕЕР ---
    def toggle_audio(self):
        if not self.selected_file: return
        if self.is_playing:
            pygame.mixer.music.stop()
            self.btn_play.configure(text="🎧 Слушать", fg_color="#17a2b8", hover_color="#138496")
            self.is_playing = False
            self.set_app_status("Воспроизведение остановлено", TEXT_COLOR)
        else:
            try:
                pygame.mixer.music.load(self.selected_file)
                pygame.mixer.music.play()
                self.btn_play.configure(text="⏹ Стоп", fg_color="#dc3545", hover_color="#c82333")
                self.is_playing = True
                self.set_app_status("🎧 Воспроизведение аудио...", ACCENT_COLOR)
                self.check_audio_playback()
            except Exception as e:
                messagebox.showerror("Ошибка аудио", f"Не удалось воспроизвести файл:\n{e}")

    def check_audio_playback(self):
        if self.is_playing and not pygame.mixer.music.get_busy():
            self.btn_play.configure(text="🎧 Слушать", fg_color="#17a2b8", hover_color="#138496")
            self.is_playing = False
            self.set_app_status("Воспроизведение завершено", TEXT_COLOR)
        elif self.is_playing:
            self.after(500, self.check_audio_playback)

    # --- ФОНОВЫЕ ПРОЦЕССЫ И МОДЕЛИ ---
    def on_model_change(self, choice):
        self.model_loaded = False
        self.btn_start.configure(state="disabled")
        self.reload_model()

    def reload_model(self):
        self.progress_bar.start()
        self.set_model_status("⏳ Загрузка параметров...", "#e5a50a")
        threading.Thread(target=self.load_models_bg, daemon=True).start()

    def load_models_bg(self):
        try:
            model_path = self.model_var.get()
            if model_path != "openai/whisper-small":
                model_path = os.path.join(MODELS_BASE_DIR, model_path)

            self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',
                                                        force_reload=False, trust_repo=True)

            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path, device_map="cuda:0",
                                                                         torch_dtype=torch.float16)
            self.model.config.forced_decoder_ids = None

            self.model_loaded = True
            self.set_model_status("● Модель активна", ACCENT_COLOR)
            self.after(0, self.progress_bar.stop)

            if self.selected_file:
                self.after(0, lambda: self.btn_start.configure(state="normal"))
        except Exception as e:
            self.after(0, self.progress_bar.stop)
            self.set_model_status("● Ошибка ядра", "#dc3545")
            self.set_app_status(f"Сбой загрузки: {e}", "#dc3545")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.m4a *.mp3 *.ogg")])
        if file_path:
            if self.is_playing:
                self.toggle_audio()

            self.selected_file = file_path
            filename = os.path.basename(file_path)

            name, ext = os.path.splitext(filename)
            display_name = f"{name[:15]}...{ext}" if len(filename) > 20 else filename

            self.file_label.configure(text=display_name)
            self.btn_play.configure(state="normal")

            if self.model_loaded:
                self.btn_start.configure(state="normal")

            self.set_app_status(f"Загружен: {display_name}", TEXT_COLOR)

    def copy_to_clipboard(self):
        text = self.textbox.get("0.0", "end").strip()
        if text and not text.startswith("Ожидание"):
            self.clipboard_clear()
            self.clipboard_append(text)
            self.set_app_status("Скопировано в буфер", ACCENT_COLOR)

    def log(self, text):
        self.after(0, lambda: self.textbox.insert("end", text + "\n"))
        self.after(0, lambda: self.textbox.see("end"))

    def start_transcription(self):
        if self.is_playing:
            self.toggle_audio()

        self.btn_start.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.textbox.delete("0.0", "end")

        self.set_app_status("Обработка аудиопотока...", "#17a2b8")
        self.set_model_status("● Нейросеть вычисляет", "#17a2b8")
        self.progress_bar.start()

        threading.Thread(target=self.process_audio, daemon=True).start()

    # --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВЫВОДА СЛОВ ---
    def _insert_timestamp(self, time_str):
        self.textbox.insert("end", time_str + " ", "timestamp")

    def _insert_word(self, word, tag):
        self.textbox.insert("end", word, tag)
        self.textbox.see("end")

    def _insert_newline(self):
        self.textbox.insert("end", "\n")

    def process_audio(self):
        start_time = time.time()
        temp_wav = "temp_processing.wav"

        try:
            command = [FFMPEG_EXE, "-y", "-i", self.selected_file, "-ar", "16000", "-ac", "1", "-loglevel", "error",
                       temp_wav]
            subprocess.run(command, check=True)

            get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = self.utils
            wav = read_audio(temp_wav, sampling_rate=16000)
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                sampling_rate=16000,
                threshold=0.3,  # КРИТИЧНО: Заставляем алгоритм верить даже тихому голосу (снизили с базовых 0.5)
                min_silence_duration_ms=500,  # Оставляем полсекунды для разделения фраз
                speech_pad_ms=300  # Даем чуть больше запаса тишины по краям
            )

            if not speech_timestamps:
                self.log("— Речь не обнаружена —")
                self.set_app_status("Сигнал отсутствует", "#ffc107")
                return

            final_text_for_db = []

            for vad_ts in speech_timestamps:
                global_chunk_start_sec = vad_ts['start'] / 16000.0
                chunk_numpy = wav[vad_ts['start']:vad_ts['end']].numpy()

                # 1. Готовим тензоры с точным совпадением типов (float16)
                input_features = self.processor(
                    chunk_numpy,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(self.model.device, dtype=torch.float16)

                # 2. Генерируем текст с параметрами подавления галлюцинаций
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_features,
                        num_beams=4,  # Чуть ускорим поиск (4 вместо 5)
                        output_scores=True,
                        return_dict_in_generate=True,
                        language="english",
                        task="transcribe",
                        return_timestamps=False,
                        # УБРАЛИ no_repeat_ngram_size, чтобы диспетчер мог повторять слова пилота
                        repetition_penalty=1.1  # Смягчили штраф за повторения
                    )

                # 3. Извлекаем оценки
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True
                )

                # Получаем сгенерированные токены
                gen_len = transition_scores.shape[1]
                generated_tokens = outputs.sequences[0, -gen_len:]

                # 4. Форматируем и печатаем таймкод
                mins = int(global_chunk_start_sec // 60)
                secs = int(global_chunk_start_sec % 60)
                time_str = f"[{mins:02d}:{secs:02d}]"

                self.after(0, lambda t=time_str: self._insert_timestamp(t))

                phrase_text = f"{time_str} "
                current_word = ""
                current_scores = []

                # 5. ЦИКЛ ПО ТОКЕНАМ: Сборка и покраска слов
                for tok, score in zip(generated_tokens, transition_scores[0]):
                    tok_id = tok.item()

                    # Декодируем токен (сохраняя пробелы)
                    chunk = self.processor.decode([tok_id], skip_special_tokens=True)
                    if not chunk:
                        continue

                    # Если встретили пробел и слово уже собирается — выводим его
                    if chunk.startswith(" ") and current_word:
                        avg_score = sum(current_scores) / len(current_scores)
                        confidence = math.exp(avg_score) * 100

                        if confidence > 80:
                            tag = "conf_high"
                        elif confidence > 50:
                            tag = "conf_mid"
                        else:
                            tag = "conf_low"

                        phrase_text += current_word
                        self.after(0, lambda w=current_word, tg=tag: self._insert_word(w, tg))

                        # Начинаем собирать новое слово
                        current_word = chunk
                        current_scores = [score.item()]
                    else:
                        # Продолжаем собирать текущее слово
                        current_word += chunk
                        current_scores.append(score.item())

                # Выводим самое последнее слово
                if current_word:
                    avg_score = sum(current_scores) / len(current_scores)
                    confidence = math.exp(avg_score) * 100

                    if confidence > 80:
                        tag = "conf_high"
                    elif confidence > 50:
                        tag = "conf_mid"
                    else:
                        tag = "conf_low"

                    phrase_text += current_word
                    self.after(0, lambda w=current_word, tg=tag: self._insert_word(w, tg))

                # Перенос строки после завершения всей фразы
                self.after(0, self._insert_newline)
                final_text_for_db.append(phrase_text.strip())

            # 6. Завершение работы: время и сохранение
            elapsed_time = time.time() - start_time

            # Сохранение чистого текста в базу данных
            save_to_db(os.path.basename(self.selected_file), "\n".join(final_text_for_db))

            # Вывод времени анализа только в статус-бар
            self.set_app_status(f"Анализ завершен ({elapsed_time:.1f} сек)", ACCENT_COLOR)

        except Exception as e:
            self.log(f"[Системная ошибка]: {e}")
            self.set_app_status("Ошибка обработки", "#dc3545")
        finally:
            self.after(0, self.progress_bar.stop)
            self.set_model_status("● Модель активна", ACCENT_COLOR)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            self.after(0, lambda: self.btn_start.configure(state="normal"))
            self.after(0, lambda: self.btn_select.configure(state="normal"))

    # --- ИСТОРИЯ И ПОИСК ---
    def delete_record_ui(self, rec_id, card_frame):
        delete_from_db(rec_id)
        card_frame.destroy()

    def view_full_text(self, fname, txt):
        view_win = ctk.CTkToplevel(self)
        view_win.title("Архив")
        view_win.geometry("600x450")
        view_win.resizable(False, False)
        view_win.configure(fg_color=BG_COLOR)
        view_win.grab_set()

        frame = ctk.CTkFrame(view_win, fg_color=PANEL_COLOR, corner_radius=12)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        name, ext = os.path.splitext(fname)
        display_fname = f"{name[:30]}...{ext}" if len(fname) > 35 else fname

        lbl = ctk.CTkLabel(frame, text=display_fname, font=ctk.CTkFont(weight="bold", size=14))
        lbl.pack(pady=(15, 5))

        tb = ctk.CTkTextbox(frame, font=ctk.CTkFont(family="Consolas", size=14), wrap="word", fg_color=TEXTBOX_BG,
                            corner_radius=8)
        tb.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        tb.insert("0.0", txt)
        tb.configure(state="disabled")

    def refresh_history_list(self, scroll_frame, search_query=""):
        for widget in scroll_frame.winfo_children():
            widget.destroy()

        records = get_history(search_query)
        if not records:
            ctk.CTkLabel(scroll_frame, text="Ничего не найдено", text_color="#666", font=ctk.CTkFont(size=14)).pack(
                pady=50)
            return

        for row in records:
            rec_id, fname, txt, tstamp = row
            name, ext = os.path.splitext(fname)
            display_fname = f"{name[:25]}...{ext}" if len(fname) > 30 else fname

            card = ctk.CTkFrame(scroll_frame, fg_color=PANEL_COLOR, corner_radius=10)
            card.pack(fill="x", pady=6, padx=5)

            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=15, pady=(12, 5))

            header = ctk.CTkLabel(header_frame, text=f"{display_fname}", font=ctk.CTkFont(weight="bold", size=13))
            header.pack(side="left")

            date_lbl = ctk.CTkLabel(header_frame, text=tstamp[:16], text_color="#666", font=ctk.CTkFont(size=11))
            date_lbl.pack(side="left", padx=10)

            btn_del = ctk.CTkButton(header_frame, text="Удалить", width=60, height=24, fg_color="transparent",
                                    text_color="#dc3545", hover_color="#2b1a1a",
                                    command=lambda r=rec_id, c=card: self.delete_record_ui(r, c))
            btn_del.pack(side="right")

            btn_view = ctk.CTkButton(header_frame, text="Открыть", width=70, height=24, fg_color=SECONDARY_BTN,
                                     hover_color=SECONDARY_HOVER,
                                     command=lambda f=fname, t=txt: self.view_full_text(f, t))
            btn_view.pack(side="right", padx=10)

            snippet = txt.replace("\n", " ")
            snippet = snippet[:85] + "..." if len(snippet) > 85 else snippet

            text_lbl = ctk.CTkLabel(card, text=snippet, text_color="#888", justify="left", font=ctk.CTkFont(size=12))
            text_lbl.pack(fill="x", padx=15, pady=(0, 12), anchor="w")

    def show_database(self):
        db_window = ctk.CTkToplevel(self)
        db_window.title("База данных")
        db_window.geometry("750x550")
        db_window.resizable(False, False)
        db_window.configure(fg_color=BG_COLOR)
        db_window.grab_set()

        search_frame = ctk.CTkFrame(db_window, fg_color="transparent")
        search_frame.pack(fill="x", padx=15, pady=(15, 0))

        search_entry = ctk.CTkEntry(search_frame, placeholder_text="🔍 Поиск по имени файла или тексту...", width=400)
        search_entry.pack(side="left", expand=True, fill="x", padx=(0, 10))

        scroll_frame = ctk.CTkScrollableFrame(db_window, fg_color="transparent")
        scroll_frame.pack(padx=10, pady=10, fill="both", expand=True)

        def on_search(event):
            self.refresh_history_list(scroll_frame, search_entry.get())

        search_entry.bind("<KeyRelease>", on_search)
        self.refresh_history_list(scroll_frame)


if __name__ == "__main__":
    app = AviationApp()
    app.mainloop()