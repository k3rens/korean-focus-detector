import torch
import whisper
import parselmouth
from fastapi import FastAPI, UploadFile
import numpy as np
import tempfile
import os
import gc

app = FastAPI()

# Load the smallest model
model = whisper.load_model("tiny", device="cpu")

@app.post("/process_audio/")
async def process_audio(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcribe
    result = model.transcribe(tmp_path)

    # Extract word-level pitch
    snd = parselmouth.Sound(tmp_path)
    pitch = snd.to_pitch()
    emphasized_words = []

    for segment in result["segments"]:
        for w in segment["words"]:
            start, end = w["start"], w["end"]
            f0_values = [
                pitch.get_value_at_time(t)
                for t in np.arange(start, end, 0.01)
            ]
            mean_f0 = np.nanmean(f0_values)
            if mean_f0 > 200:  # example threshold
                emphasized_words.append(f"*{w['word']}*")
            else:
                emphasized_words.append(w["word"])

    os.remove(tmp_path)
    del snd, pitch
    gc.collect()

    return {"emphasized_text": " ".join(emphasized_words)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # 1. Load audio into Whisper
    audio_buffer = io.BytesIO(audio_bytes)
    result = model.transcribe(audio_buffer, task="translate")
    words = [w["text"].strip() for seg in result["segments"] for w in [seg]]  # word list
    # 2. Extract Praat pitch values per word
    snd = parselmouth.Sound(audio_bytes)
    pitch_obj = snd.to_pitch()
    pitch_values = []
    for w in words:
        # Simple mapping: divide total frames evenly per word
        total_frames = pitch_obj.get_number_of_frames()
        frames_per_word = max(total_frames // len(words), 1)
        start_frame = len(pitch_values)
        end_frame = min(start_frame + frames_per_word, total_frames)
        frame_pitches = [pitch_obj.get_value_at_time(pitch_obj.get_time_from_frame_number(f))
                         for f in range(start_frame, end_frame)]
        mean_pitch = np.mean([p for p in frame_pitches if p is not None and p > 0] or [0])
        pitch_values.append(mean_pitch)

    # 3. Compute all thresholds emphasis
    thresholds = [73.35205304, 70.144100855, 66.93614867, 63.728196485,
                  60.5202443, 57.312292115, 54.10433993]
    results = compute_emphasis_all_thresholds(words, pitch_values, thresholds)

    return {"results": results}
