# app.py - Audio -> Whisper Transcription (KO) & Whisper Translation (EN)

import os
import time
import numpy as np
import soundfile as sf
import parselmouth
import sys
import re
import asyncio

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper 

# ------------------------
# Global Thresholds (Based on ULTIMATE Stage)
# ------------------------
T_START = 73.35205304
T_END = 54.10433993
NUM_THRESHOLDS = 7
STEP = (T_START - T_END) / (NUM_THRESHOLDS - 1)
SEVEN_THRESHOLDS = [T_START - i * STEP for i in range(NUM_THRESHOLDS)]

app = FastAPI()

# Enhanced CORS setup to prevent "Failed to fetch" errors in browser environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load 'base' model for better accuracy. 
# Note: 'base' is significantly more reliable for translation than 'tiny'
print("Loading Whisper 'base' model...")
model = whisper.load_model("base", device="cpu") 
print("Model loaded.")

async def analyze_and_compute(filename):
    temp_pcm_filename = filename.replace('.webm', '_pcm.wav') 
    try:
        # Load audio using Whisper's robust utility
        audio_data = whisper.load_audio(filename)
        sf.write(temp_pcm_filename, audio_data, samplerate=16000, subtype='PCM_16')
        
        # 1. Whisper Transcription (Korean)
        # word_timestamps=True is essential for mapping pitch to specific words
        result_ko = model.transcribe(
            temp_pcm_filename, 
            language="ko", 
            word_timestamps=True, 
            fp16=False
        )
        ko_text = result_ko.get("text", "").strip()
        
        # 2. Whisper Translation (English)
        # Using the built-in translation task with temperature 0 for stability
        result_en = model.transcribe(
            temp_pcm_filename, 
            task="translate", 
            fp16=False,
            temperature=0.0
        )
        base_english = result_en.get("text", "").strip()

        # 3. Acoustic Pitch Extraction
        snd = parselmouth.Sound(temp_pcm_filename)
        pitch = snd.to_pitch()
        
        words, word_stamps = [], []
        for seg in result_ko.get("segments", []):
            if seg.get("words"):
                for w in seg["words"]:
                    if w.get("word", "").strip():
                        words.append(w["word"].strip())
                        word_stamps.append((float(w["start"]), float(w["end"])))

        if not words:
            return {"error": "No speech detected. Please speak clearly into the microphone."}

        pitch_values = []
        for (start, end) in word_stamps:
            times = np.arange(start, end, 0.01)
            vals = [pitch.get_value_at_time(t) for t in times]
            filtered = [v for v in vals if v and v > 0] 
            pitch_values.append(float(np.mean(filtered)) if filtered else 0.0)

        # 4. Compute Emphasis across thresholds
        results = []
        for t in SEVEN_THRESHOLDS:
            ratios = []
            candidates = []
            for i in range(len(pitch_values) - 1):
                p_curr, p_next = pitch_values[i], pitch_values[i+1]
                ratio = (p_next / p_curr) * 100.0 if p_curr > 0 and p_next > 0 else 100.0
                ratios.append(ratio)
                if ratio < t:
                    candidates.append(i)
            
            emphasized_ko_list = words.copy()
            focus_word_ko = ""
            if candidates:
                idx = min(candidates, key=lambda i: ratios[i])
                focus_word_ko = words[idx]
                emphasized_ko_list[idx] = f"*{words[idx]}*"
            
            en_display = base_english
            if focus_word_ko:
                en_display = f"<i>{base_english}</i> (Focus: {focus_word_ko})"

            processed_table = []
            for i, w in enumerate(words):
                is_focus = (candidates and i == min(candidates, key=lambda k: ratios[k]))
                processed_table.append({
                    "word": w, 
                    "pitch": round(pitch_values[i], 2), 
                    "emphasis": "Focus" if is_focus else "Neutral"
                })

            results.append({
                "threshold": round(t, 2),
                "ko_text": " ".join(emphasized_ko_list),
                "en_text": en_display,
                "table_data": processed_table
            })

        return {
            "all_results": results,
            "base_english": base_english
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_pcm_filename):
            try:
                os.remove(temp_pcm_filename)
            except:
                pass

@app.post("/translate/")
async def handle_request(file: UploadFile = File(...)):
    temp_filename = f"upload_{time.time()}.webm" 
    try:
        content = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(content)
        # Using await for the processing task
        data = await analyze_and_compute(temp_filename)
        return JSONResponse(data)
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

@app.get("/")
async def root():
    return HTMLResponse("Server is active. Listening on port 8000.")

if __name__ == "__main__":
    import uvicorn
    # explicitly binding to 127.0.0.1 for local dev
    uvicorn.run(app, host="127.0.0.1", port=8000)
