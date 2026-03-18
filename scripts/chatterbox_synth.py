#!/usr/bin/env python3
"""
Chatterbox TTS synthesis helper for livescribe.

Reads sentences from stdin (one per line, JSON-encoded), synthesizes audio,
and writes raw f32 PCM to stdout with a simple framing protocol:
  - 4 bytes: sample rate as little-endian u32 (first chunk only)
  - For each sentence:
    - 4 bytes: audio length in samples as little-endian u32
    - N*4 bytes: f32 samples in little-endian

Usage:
  echo '{"text":"Hello world"}' | python chatterbox_synth.py [--voice ref.wav] [--model turbo]
  Or pipe multiple JSON lines for batch processing.

Requires: pip install chatterbox-tts
"""

import sys
import json
import struct
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS sidecar")
    parser.add_argument("--voice", type=str, default=None,
                        help="Path to reference audio for voice cloning")
    parser.add_argument("--model", type=str, default="turbo",
                        choices=["turbo", "multilingual", "original"],
                        help="Chatterbox model variant (default: turbo)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (cuda, cpu, mps)")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Exaggeration parameter for original model")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="CFG weight for original model")
    args = parser.parse_args()

    # Detect device
    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sys.stderr.write(f"[chatterbox] Loading model '{args.model}' on {device}...\n")

    # Load model
    if args.model == "turbo":
        from chatterbox.tts import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device=device)
    elif args.model == "multilingual":
        from chatterbox.tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)

    sample_rate = model.sr
    sys.stderr.write(f"[chatterbox] Model loaded. Sample rate: {sample_rate}Hz\n")

    # Write sample rate header
    sys.stdout.buffer.write(struct.pack("<I", int(sample_rate)))
    sys.stdout.buffer.flush()

    # Process sentences from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"[chatterbox] Invalid JSON: {line}\n")
            # Write zero-length chunk to signal skip
            sys.stdout.buffer.write(struct.pack("<I", 0))
            sys.stdout.buffer.flush()
            continue

        text = msg.get("text", "")
        if not text:
            sys.stdout.buffer.write(struct.pack("<I", 0))
            sys.stdout.buffer.flush()
            continue

        try:
            # Generate audio
            generate_kwargs = {"text": text}
            if args.voice:
                generate_kwargs["audio_prompt_path"] = args.voice

            if args.model == "original":
                generate_kwargs["exaggeration"] = args.exaggeration
                generate_kwargs["cfg_weight"] = args.cfg_weight

            wav = model.generate(**generate_kwargs)

            # Convert to numpy f32
            audio = wav.squeeze().cpu().numpy().astype(np.float32)

            # Normalize to [-1, 1]
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.95

            # Write framed audio
            sys.stdout.buffer.write(struct.pack("<I", len(audio)))
            sys.stdout.buffer.write(audio.tobytes())
            sys.stdout.buffer.flush()

            sys.stderr.write(f"[chatterbox] Synthesized {len(audio)} samples\n")

        except Exception as e:
            sys.stderr.write(f"[chatterbox] Error: {e}\n")
            sys.stdout.buffer.write(struct.pack("<I", 0))
            sys.stdout.buffer.flush()

    sys.stderr.write("[chatterbox] Done.\n")

if __name__ == "__main__":
    main()
