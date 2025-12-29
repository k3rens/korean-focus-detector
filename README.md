*Korean Focus-Detecting Translation/Transcription Tool*

This project is a phonetic analysis tool designed to detect prosodic focus in spoken Korean sentences. It uses real-time acoustic analysis to identify the "focus word" based on the principle of Post-Focal Compression (PFC).

(View demo at https://k3rens-koreanfocusdetectordemo.static.hf.space/index.html
Note: the demo UI is significantly different from the HTML/CSS files in this repository, as this project aims to demonstrate the full seven thresholds, while the demo only highlights the most accurate one.)

A. How to Run Locally

To use this tool, you will need to run the Python backend and open the HTML frontend simultaneously.

1. Prerequisites

Ensure you have Python 3.8+ installed. You also need FFmpeg installed on your system for audio processing (Whisper requirement).

2. Installation

Clone this repository and install the necessary Python libraries:

   a) Clone the repo

    git clone [https://github.com/k3rens/korean-focus-detector.git](https://github.com/k3rens/korean-focus-detector.git)
    cd korean-focus-detector

  b) Install dependencies

    pip install fastapi uvicorn openai-whisper python-multipart soundfile praat-parselmouth numpy


3. Usage

Start the Backend:
Run the FastAPI server:

python app.py


Note: On the first run, it will download the Whisper 'base' model (~140MB).

Open the Frontend:
Simply open emphasis_detector.html in any modern web browser (Chrome or Edge recommended).

Analyze:
Click "Start Recording," speak a Korean sentence with natural emphasis, and click "Stop." The tool will generate seven tables based on different pitch ratio thresholds.

B. Linguistic Background

In Korean phonology, prominence is often marked by a significant pitch drop in the word following the emphasized word. This application calculates the Pitch Ratio:

$$\frac{\text{Pitch}_{(\text{following word})}}{\text{Pitch}_{(\text{focus word})}} \times 100\%$$

The system evaluates this ratio against seven thresholds derived from the research of Chung & Kenstowicz (1997). Our testing shows that a threshold of 73.35% is the most reliable indicator of focus across various native speakers.

C. Technology Stack

Backend: FastAPI (Python)

Transcription/Translation: OpenAI Whisper (Base Model)

Note: The files in this repository follow a (Korean Speech) to (Korean Text) to (English Text) process. The first step of transcription is fairly accurate with Whisper. However, the (Text) to (Text) translation can be inaccurate depending on the usage. Swap out the model if another text-to-text translation model is more appropriate for your purpose in seeking out focus-detection. 

Acoustic Analysis: Parselmouth (Praat for Python)

Frontend: HTML5, Tailwind CSS, JavaScript (MediaRecorder API)

Created for linguistic research and educational purposes! Have fun!
