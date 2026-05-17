import os
import tempfile
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import httpx
import pandas as pd
import librosa
import io
import numpy as np
from Services.FeatureExtractionAndMLService import predict_dialect, generate_inference_visualization_data
from Services.AudioProcessingService import compute_mel_spectrogram, compute_spectrogram

from pydantic import BaseModel

#py -m uvicorn endpoints:app --reload
app = FastAPI(title="Arabic Dialects API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GPU_BACKEND_URL = "https://arbitrate-motivator-impale.ngrok-free.dev"

_ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
 
 
def _load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(file_bytes)
    y, sr = librosa.load(buf, sr=None, mono=True)
    return y, sr
 
 
def _validate_and_load(file: UploadFile, raw: bytes) -> tuple[np.ndarray, int]:
    if not (
        (file.content_type and file.content_type.startswith("audio/"))
        or (file.filename and file.filename.endswith(_ALLOWED_EXTENSIONS))
    ):
        raise HTTPException(status_code=400, detail="Unsupported file type. Send an audio file.")
    try:
        return _load_audio(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

class TTSRequest(BaseModel):
    text: str
    dialect: str = "Egyptian"
    speaker_id: int = 1

class TranslationRequest(BaseModel):
    text: str
    original_dialect: str
    target_dialect: str

# --- 1. Load the Model and Encoder ---
try:
    print("Loading pipeline and encoder...")
    pipeline = joblib.load("MLModels/dialect_classifier_pipeline.pkl")
    le = joblib.load("MLModels/dialect_label_encoder.pkl")
except FileNotFoundError:
    print("Warning: Model files not found. Make sure .pkl files are in the same directory.")

train_df = pd.read_csv("Data/Training_Data.csv")

y_train = train_df["dialect"]
X_train = train_df.drop(["dialect"], axis=1)

# --- 3. The API Endpoint ---
@app.post("/predict")
async def classify_audio(file: UploadFile = File(...)):
    # Basic file extension validation
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        try:
            contents = await file.read()
            temp_audio.write(contents)
            temp_audio_path = temp_audio.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        # Call your custom predict function
        result = predict_dialect(temp_audio_path, pipeline, le)
        
        # Handle the case where your function returns an error string
        if isinstance(result, str) and result.startswith("Error"):
            raise HTTPException(status_code=400, detail=result)

        # Unpack the results (assuming your function returns a tuple)
        predicted_label, confidence, features= result

        return {
            "filename": file.filename,
            "predicted_dialect": predicted_label,
            "confidence": confidence,
            "features": dict(zip(X_train.columns, features))  # list → named dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.post("/transcribe")
async def proxy_transcribe(file: UploadFile = File(...)):
    """Forwards the audio file to the GPU server for transcription."""
    file_bytes = await file.read()
    
    # httpx expects files in the format: {"field_name": (filename, file_bytes, content_type)}
    files = {"file": (file.filename, file_bytes, file.content_type)}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{GPU_BACKEND_URL}/transcribe", files=files)
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to GPU backend: {str(e)}")

@app.post("/tts")
async def proxy_tts(request: TTSRequest):
    """Forwards the text and gets an audio file back from the GPU server."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{GPU_BACKEND_URL}/tts", json=request.dict())
            response.raise_for_status()
            
            # Since the GPU server returns an audio file (FileResponse), 
            # we pipe those raw bytes directly back to the frontend.
            return Response(content=response.content, media_type="audio/wav")
            
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to GPU backend: {str(e)}")

@app.post("/translate")
async def proxy_translate(request: TranslationRequest):
    """Forwards text for translation."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{GPU_BACKEND_URL}/translate", json=request.dict())
            response.raise_for_status()
            return JSONResponse(content=response.json())
            
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to GPU backend: {str(e)}")

@app.post("/visualize_prediction")
async def visualize_prediction(features: dict):
    """
    Takes the extracted features of a newly uploaded audio file, 
    and returns the 3D map data showing exactly where it lands.
    """
    try:
        # 1. Convert the incoming JSON features into a DataFrame
        X_test_sample = pd.DataFrame([features])

        # 2. ALIGNMENT: Ensure columns match X_train exactly
        # If the incoming audio is missing features that were in the training set, pad with 0
        missing_cols = set(X_train.columns) - set(X_test_sample.columns)
        for col in missing_cols:
            X_test_sample[col] = 0.0
            
        # Reorder the columns so they are in the exact same sequence as the scaler/LDA expects
        X_test_sample = X_test_sample[X_train.columns]

        # 3. Call the visualization function, passing the feature names
        computed_data = generate_inference_visualization_data(
            X_train=X_train, 
            y_train_strings=y_train, 
            X_test_sample=X_test_sample,
            feature_names=X_train.columns.tolist() # <--- Added this so the JSON returns actual names!
        )
        
        return JSONResponse(content=computed_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.post("/spectrogram")
async def get_spectrogram(file: UploadFile = File(...)):
    """Return a linear-frequency STFT spectrogram as a PNG."""
    raw = await file.read()
    y, sr = _validate_and_load(file, raw)
    buf = compute_spectrogram(y, sr)
    return StreamingResponse(buf, media_type="image/png")
 
 
@app.post("/mel-spectrogram")
async def get_mel_spectrogram(
    file: UploadFile = File(...),
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
):
    """Return a Mel-scale spectrogram as a PNG.
 
    Query params:
      - n_mels (default 128)
      - fmin   (default 0 Hz)
      - fmax   (default sr/2)
    """
    raw = await file.read()
    y, sr = _validate_and_load(file, raw)
    buf = compute_mel_spectrogram(y, sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return StreamingResponse(buf, media_type="image/png")