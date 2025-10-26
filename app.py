import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

import whisper
import numpy as np
import parselmouth
from emphasis import compute_emphasis_all_thresholds

model = whisper.load_model("base")

print("Whisper model loaded.")

app = FastAPI(title="Korean Focus Translator (Post-focal ratio)")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
