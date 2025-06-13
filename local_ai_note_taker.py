# local_ai_note_taker.py

"""
LocalNoteAI Prototype: Local AI Note-Taking System for Google Meet

Features:
- Records audio locally (until Enter is pressed)
- Transcribes audio using Whisper
- Summarizes transcript using local LLM
- Embeds and stores notes for semantic search
- Command-line UI for choosing actions
- Export notes to PDF
"""

import os
import datetime
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from transformers import pipeline
import faiss
import numpy as np
import sqlite3
import threading
import queue
from fpdf import FPDF

# Paths
RECORD_DIR = "recordings"
EXPORT_DIR = "exports"
DB_PATH = "local_notes.db"
EMBEDDING_DIM = 384  # for all-MiniLM-L6-v2

# Initialize transcription and summarization
whisper_model = WhisperModel("medium", compute_type="int8")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Create directories
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Create DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    filename TEXT,
    transcript TEXT,
    summary TEXT
)
""")
conn.commit()

# FAISS index setup
index = faiss.IndexFlatL2(EMBEDDING_DIM)

# Placeholder embedding function (use SentenceTransformers locally)
def embed(text):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text])[0]

# Record audio until Enter is pressed
def record_audio_until_enter():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"meeting_{timestamp}.wav"
    filepath = os.path.join(RECORD_DIR, filename)
    print(f"Recording started. Press Enter to stop...")

    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(indata.copy())

    def input_thread():
        input()  # Wait for Enter key
        q.put(None)

    threading.Thread(target=input_thread, daemon=True).start()

    with sf.SoundFile(filepath, mode='x', samplerate=16000, channels=1, subtype='PCM_16') as file:
        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            while True:
                data = q.get()
                if data is None:
                    break
                file.write(data)

    print(f"Recording saved to {filepath}")
    return filepath

# Transcribe
def transcribe_audio(filepath):
    segments, info = whisper_model.transcribe(filepath, beam_size=5, vad_filter=True)
    speaker_transcript = []
    speaker_id = 0
    last_end = 0.0
    for segment in segments:
        if segment.start - last_end > 1.5:
            speaker_id += 1
        speaker_transcript.append(f"Speaker {speaker_id + 1}: {segment.text}")
        last_end = segment.end
    full_text = "\n".join(speaker_transcript)
    return full_text

# Summarize
def summarize_text(text):
    tagged_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = "\n".join([s["summary_text"] for s in summarizer(tagged_chunks)])
    return summary

# Save note
def save_note(filename, transcript, summary):
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("INSERT INTO notes (timestamp, filename, transcript, summary) VALUES (?, ?, ?, ?)",
                   (timestamp, filename, transcript, summary))
    conn.commit()
    vec = embed(summary)
    index.add(np.array([vec]))
    print("Note saved and embedded.")

# View all summaries
def view_summaries():
    cursor.execute("SELECT timestamp, summary FROM notes ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    for ts, summary in rows:
        print(f"\n[{ts}]\n{summary}\n")

# Export notes to PDF
def export_notes_to_pdf():
    cursor.execute("SELECT timestamp, filename, transcript, summary FROM notes ORDER BY timestamp DESC")
    notes = cursor.fetchall()
    for idx, (timestamp, filename, transcript, summary) in enumerate(notes):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Meeting Notes - {timestamp}", ln=True, align='L')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Summary:\n" + summary)
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Transcript:\n" + transcript)
        base_name = os.path.splitext(os.path.basename(filename))[0].replace(":", "-")
        safe_timestamp = timestamp.replace(":", "-").replace("T", "_").split(".")[0]
        export_filename = os.path.join(EXPORT_DIR, f"{base_name}_{safe_timestamp}.pdf")
        pdf.output(export_filename)
        print(f"Exported: {export_filename}")

# CLI Menu
if __name__ == "__main__":
    while True:
        print("\nWelcome to LocalNoteAI 🧠")
        print("1. Record new meeting")
        print("2. View all summaries")
        print("3. Export notes to PDF")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            path = record_audio_until_enter()
            text = transcribe_audio(path)
            summary = summarize_text(text)
            save_note(path, text, summary)
            print("Done.")

        elif choice == "2":
            view_summaries()

        elif choice == "3":
            export_notes_to_pdf()

        elif choice == "4":
            print("Exiting. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")
