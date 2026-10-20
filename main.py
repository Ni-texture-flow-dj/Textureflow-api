from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import io

app = FastAPI(title="API Textureflow DJ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API Textureflow opérationnelle"}

@app.post("/analyze-texture")
async def analyze_texture(file: UploadFile = File(...)):
    contents = await file.read()
    y, sr = librosa.load(io.BytesIO(contents), sr=None)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    texture_type = "Kick / Low-End" if spectral_centroid < 2000 else "Synthé / High-End / Texture"

    return {
        "filename": file.filename,
        "tempo": float(tempo),
        "spectral_centroid": spectral_centroid,
        "texture_type": texture_type
    }
  
