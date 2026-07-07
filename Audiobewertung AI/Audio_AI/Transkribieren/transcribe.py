import sys
import os
import whisper
import tkinter as tk
from tkinter import filedialog

def pick_video_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Video auswählen",
        filetypes=[("Video-Dateien", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"), ("Alle Dateien", "*.*")]
    )
    root.destroy()
    return path

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def transcribe(video_path: str) -> list:
    print("Lade Whisper-Modell...")
    model = whisper.load_model("base")
    print(f"Transkribiere: {video_path}")
    result = model.transcribe(video_path, word_timestamps=True)
    words = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"],
            })
    return words

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = pick_video_file()

    if not video_path:
        print("Kein Video ausgewählt. Abbruch.")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"Datei nicht gefunden: {video_path}")
        sys.exit(1)

    words = transcribe(video_path)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Transkripte")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, base_name + "_transkript.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'Wort':<30} {'Start':>12} {'Ende':>12}\n")
        f.write("-" * 56 + "\n")
        for w in words:
            f.write(f"{w['word']:<30} {format_time(w['start']):>12} {format_time(w['end']):>12}\n")

    print(f"\nTranskript gespeichert: {output_path}")
    print(f"\n{'Wort':<30} {'Start':>12} {'Ende':>12}")
    print("-" * 56)
    for w in words:
        print(f"{w['word']:<30} {format_time(w['start']):>12} {format_time(w['end']):>12}")

if __name__ == "__main__":
    main()
