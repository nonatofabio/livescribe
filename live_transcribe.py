#!/usr/bin/env python3
"""
Simple Live Audio Transcription Script
Records audio from microphone and transcribes in real-time using Whisper
"""

import pyaudio
import wave
import whisper
import numpy as np
import sys
import os
from datetime import datetime
import argparse

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 8  # Process audio in 8-second chunks for better context

def setup_audio():
    """Initialize PyAudio"""
    p = pyaudio.PyAudio()
    
    # List available input devices
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    
    return p

def record_chunk(stream, duration=RECORD_SECONDS):
    """Record a chunk of audio"""
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Error reading audio: {e}")
            break
    return frames

def frames_to_audio(frames):
    """Convert audio frames to numpy array for Whisper"""
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def main():
    parser = argparse.ArgumentParser(description='Live audio transcription using Whisper')
    parser.add_argument('-o', '--output', default='transcription.txt', 
                        help='Output file for transcription (default: transcription.txt)')
    parser.add_argument('-m', '--model', default='base', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    parser.add_argument('-d', '--device', type=int, 
                        help='Audio input device index (optional)')
    parser.add_argument('-c', '--chunk-duration', type=int, default=8,
                        help='Duration of audio chunks to process in seconds (default: 8)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Live Audio Transcription")
    print("="*60)
    print(f"Output file: {args.output}")
    print(f"Whisper model: {args.model}")
    print(f"Chunk duration: {args.chunk_duration}s")
    
    # Load Whisper model
    print(f"\nLoading Whisper {args.model} model...")
    model = whisper.load_model(args.model)
    print("Model loaded!")
    
    # Setup audio
    p = setup_audio()
    
    # Select device
    device_index = args.device
    if device_index is None:
        device_index = p.get_default_input_device_info()['index']
        print(f"\nUsing default device: [{device_index}] {p.get_device_info_by_index(device_index)['name']}")
    else:
        print(f"\nUsing device: [{device_index}] {p.get_device_info_by_index(device_index)['name']}")
    
    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    
    print("\n" + "="*60)
    print("Recording and transcribing... Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Open output file
    with open(args.output, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'='*60}\n")
        f.write(f"Transcription started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        try:
            chunk_num = 0
            while True:
                chunk_num += 1
                print(f"[Chunk {chunk_num}] Recording {args.chunk_duration}s...")
                
                # Record audio chunk
                frames = record_chunk(stream, args.chunk_duration)
                
                if not frames:
                    continue
                
                # Convert to format Whisper expects
                audio_np = frames_to_audio(frames)
                
                # Transcribe with anti-hallucination parameters
                print(f"[Chunk {chunk_num}] Transcribing...")
                result = model.transcribe(
                    audio_np,
                    language='en',
                    fp16=False,
                    condition_on_previous_text=False,  # Prevents hallucination loops
                    temperature=0.0,  # More deterministic
                    compression_ratio_threshold=1.8,   # Lower = stricter repetition filter
                    logprob_threshold=-0.5,            # Higher = stricter confidence
                    no_speech_threshold=0.7            # Higher = better silence detection
                )
                text = result['text'].strip()
                
                # Detect and reject hallucination loops (repeated phrases)
                if text:
                    words = text.split()
                    if len(words) >= 6:
                        phrase_len = len(words) // 3
                        if phrase_len >= 2:
                            first_phrase = ' '.join(words[:phrase_len])
                            if text.count(first_phrase) >= 3:
                                print(f"[Chunk {chunk_num}] Hallucination detected, skipping")
                                text = ""
                
                if text:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    output_line = f"[{timestamp}] {text}\n"
                    
                    # Print to console
                    print(f"\n{output_line}")
                    
                    # Write to file
                    f.write(output_line)
                    f.flush()  # Ensure it's written immediately
                else:
                    print(f"[Chunk {chunk_num}] No speech detected\n")
                    
        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
            f.write(f"\n{'='*60}\n")
            f.write(f"Transcription ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print(f"\nTranscription saved to: {args.output}")

if __name__ == "__main__":
    main()