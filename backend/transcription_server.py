#!/usr/bin/env python3
"""
WebSocket server for live audio transcription
Provides real-time transcription via WebSocket for the Tauri frontend
"""

import asyncio
import json
import pyaudio
import whisper
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Global state
class AppState:
    model: Optional[whisper.Whisper] = None
    model_name: str = "base"
    is_recording: bool = False
    audio: Optional[pyaudio.PyAudio] = None
    stream = None
    current_device: Optional[int] = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    state.audio = pyaudio.PyAudio()
    yield
    # Cleanup
    if state.stream:
        state.stream.stop_stream()
        state.stream.close()
    if state.audio:
        state.audio.terminate()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelRequest(BaseModel):
    model: str

@app.get("/devices")
async def get_devices():
    """List available audio input devices"""
    devices = []
    for i in range(state.audio.get_device_count()):
        info = state.audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append({
                "index": i,
                "name": info['name'],
                "channels": info['maxInputChannels'],
                "sample_rate": int(info['defaultSampleRate'])
            })
    
    default_device = state.audio.get_default_input_device_info()
    return {
        "devices": devices,
        "default": default_device['index']
    }

@app.get("/models")
async def get_models():
    """List available Whisper models"""
    models = [
        {"name": "tiny", "size": "39 MB", "speed": "Fastest", "accuracy": "Good"},
        {"name": "base", "size": "74 MB", "speed": "Fast", "accuracy": "Better"},
        {"name": "small", "size": "244 MB", "speed": "Moderate", "accuracy": "Great"},
        {"name": "medium", "size": "769 MB", "speed": "Slower", "accuracy": "Excellent"},
        {"name": "large", "size": "1550 MB", "speed": "Slowest", "accuracy": "Best"},
    ]
    return {
        "models": models,
        "current": state.model_name,
        "loaded": state.model is not None
    }

@app.post("/load-model")
async def load_model(request: ModelRequest):
    """Load a Whisper model"""
    try:
        state.model_name = request.model
        state.model = whisper.load_model(request.model)
        return {"success": True, "model": request.model}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/status")
async def get_status():
    """Get current transcription status"""
    return {
        "is_recording": state.is_recording,
        "model_loaded": state.model is not None,
        "model_name": state.model_name,
        "device": state.current_device
    }

def record_chunk(stream, duration: float) -> list:
    """Record a chunk of audio"""
    frames = []
    num_chunks = int(RATE / CHUNK * duration)
    for _ in range(num_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Error reading audio: {e}")
            break
    return frames

def frames_to_audio(frames) -> np.ndarray:
    """Convert audio frames to numpy array for Whisper"""
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription"""
    await websocket.accept()
    
    try:
        # Wait for configuration
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        
        device_index = config.get("device")
        chunk_duration = config.get("chunk_duration", 8)
        language = config.get("language", "en")
        
        # Load model if not loaded
        if state.model is None:
            await websocket.send_json({
                "type": "status",
                "message": f"Loading Whisper {state.model_name} model..."
            })
            state.model = whisper.load_model(state.model_name)
            await websocket.send_json({
                "type": "status", 
                "message": "Model loaded!"
            })
        
        # Get default device if not specified
        if device_index is None:
            device_index = state.audio.get_default_input_device_info()['index']
        
        state.current_device = device_index
        
        # Open audio stream
        stream = state.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        state.stream = stream
        state.is_recording = True
        
        device_name = state.audio.get_device_info_by_index(device_index)['name']
        await websocket.send_json({
            "type": "started",
            "device": device_name,
            "device_index": device_index
        })
        
        chunk_num = 0
        while state.is_recording:
            chunk_num += 1
            
            await websocket.send_json({
                "type": "recording",
                "chunk": chunk_num,
                "duration": chunk_duration
            })
            
            # Record audio chunk in a thread to not block
            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                None, record_chunk, stream, chunk_duration
            )
            
            if not frames or not state.is_recording:
                continue
            
            await websocket.send_json({
                "type": "processing",
                "chunk": chunk_num
            })
            
            # Convert and transcribe
            audio_np = frames_to_audio(frames)
            
            # Transcribe in executor to not block
            def do_transcribe():
                return state.model.transcribe(
                    audio_np,
                    language=language,
                    fp16=False,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
            
            result = await loop.run_in_executor(None, do_transcribe)
            text = result['text'].strip()
            
            if text:
                timestamp = datetime.now().strftime('%H:%M:%S')
                await websocket.send_json({
                    "type": "transcription",
                    "text": text,
                    "timestamp": timestamp,
                    "chunk": chunk_num
                })
            else:
                await websocket.send_json({
                    "type": "silence",
                    "chunk": chunk_num
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        state.is_recording = False
        if state.stream:
            state.stream.stop_stream()
            state.stream.close()
            state.stream = None

@app.post("/stop")
async def stop_recording():
    """Stop the current recording session"""
    state.is_recording = False
    return {"success": True}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)

