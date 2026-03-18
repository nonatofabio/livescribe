<p align="center">
  <img src="assets/logo.svg" alt="Livescribe" width="480"/>
</p>

<p align="center">
  <a href="#installation"><strong>Install</strong></a> &middot;
  <a href="#listen-speech-to-text"><strong>Listen (STT)</strong></a> &middot;
  <a href="#speak-text-to-speech"><strong>Speak (TTS)</strong></a> &middot;
  <a href="#architecture"><strong>Architecture</strong></a>
</p>

<p align="center">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white"/>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-green?style=flat-square"/>
</p>

---

A fast, cross-platform audio toolkit built in Rust — **live transcription** (STT) using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and **document reading** (TTS) using [Piper](https://github.com/rhasspy/piper).

## Features

- **Listen**: Real-time speech-to-text with non-blocking pipeline, overlapping chunks, and automatic deduplication
- **Speak**: Read documents aloud (.txt, .md, .pdf) with high-quality neural TTS voices
- **Auto model download**: Whisper and Piper models downloaded from HuggingFace on first run
- **Cross-platform**: macOS (CoreAudio), Linux (ALSA), Windows (WASAPI) via cpal
- **GPU acceleration**: Optional Metal (macOS), CUDA (NVIDIA), and CoreML support for transcription

## Installation

### Prerequisites

- Rust toolchain ([rustup](https://rustup.rs/))
- C/C++ compiler (Xcode CLI Tools on macOS, gcc on Linux, MSVC on Windows)
- `espeak-ng` for TTS phonemization:
  ```bash
  # macOS
  brew install espeak-ng
  # Debian/Ubuntu
  sudo apt install espeak-ng
  ```
- On Linux: `libasound2-dev` (Debian/Ubuntu) or `alsa-lib-devel` (Fedora)

### Build

```bash
cargo build --release

# macOS with Metal GPU acceleration (for listen)
cargo build --release --features metal
```

Install to PATH:
```bash
cargo install --path .
```

---

## Listen (Speech-to-Text)

Real-time microphone transcription using Whisper.

```bash
# Start transcribing (auto-downloads distil-large-v3 on first run, ~1.5 GB)
livescribe listen

# Use a specific model and output file
livescribe listen --model small --output meeting.txt

# Use a specific microphone
livescribe listen --device 2

# List available input devices
livescribe listen --list-devices
```

### Listen Options

```
  -o, --output <FILE>           Output file [default: transcription.txt]
  -m, --model <NAME|PATH>       Whisper model [default: distil-large-v3]
  -d, --device <INDEX>          Audio input device index
  -c, --chunk-duration <SECS>   Chunk size [default: 8]
      --overlap <SECS>          Overlap between chunks [default: 2]
  -l, --language <CODE>         Language [default: en]
  -t, --threads <N>             Whisper inference threads
      --list-devices            List input devices and exit
```

### Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 75 MB | Fastest | Good |
| `base` | 142 MB | Fast | Better |
| `small` | 466 MB | Moderate | Great |
| `medium` | 1.5 GB | Slower | Excellent |
| `large-v3` | 3.1 GB | Slowest | Best |
| **`distil-large-v3`** | **1.5 GB** | **Fast** | **Near-best** |

---

## Speak (Text-to-Speech)

Read documents aloud using Piper neural TTS.

```bash
# Read a text file (auto-downloads voice model on first run, ~63 MB)
livescribe speak document.txt

# Read a Markdown file with a different voice
livescribe speak notes.md --voice en_US-lessac-medium

# Read a PDF and save to WAV
livescribe speak paper.pdf --save output.wav

# Save without playing
livescribe speak doc.txt --save output.wav --no-play

# Adjust speech speed (2x faster)
livescribe speak doc.txt --speed 2.0

# List available voices
livescribe speak --list-voices

# List available output devices
livescribe speak --list-devices
```

### Speak Options

```
  <FILE>                    Document to read (.txt, .md, .pdf)
  -v, --voice <NAME>        Piper voice [default: en_US-amy-medium]
  -s, --save <PATH>         Save audio to WAV file
      --speed <FLOAT>       Speech speed multiplier [default: 1.0]
  -d, --device <INDEX>      Audio output device index
      --list-voices         List available voices
      --list-devices        List output devices
      --no-play             Don't play audio (use with --save)
```

### Piper Voices

| Voice | Language | Gender | Quality |
|-------|----------|--------|---------|
| `en_US-amy-medium` | US English | Female | Medium |
| `en_US-lessac-medium` | US English | Male | Medium |
| `en_US-lessac-high` | US English | Male | High |
| `en_US-ryan-medium` | US English | Male | Medium |
| `en_US-joe-medium` | US English | Male | Medium |
| `en_GB-alba-medium` | British English | Female | Medium |
| `en_GB-jenny_dioco-medium` | British English | Female | Medium |

### Supported Document Formats

- **`.txt`** — Plain text, read as-is
- **`.md`** — Markdown with formatting stripped (code blocks skipped)
- **`.pdf`** — PDF text extraction (text-based PDFs only)

---

## Architecture

### Listen Pipeline (3 threads)
```
[Audio Thread] → [Transcription Thread] → [Output / Main Thread]
    cpal            whisper.cpp               file + stdout
 (never stops)     (CPU-bound)              (dedup + write)
```

### Speak Pipeline (2 threads)
```
[Synthesis Thread] → [Playback / Main Thread]
  espeak-ng + Piper       cpal output stream
   (CPU-bound)           (resample + play)
```

Both pipelines use bounded crossbeam channels with backpressure. Recording never pauses during transcription. Ctrl+C triggers graceful shutdown with no data loss.

## Capture Internal Audio (macOS)

To transcribe system audio (video calls, browser), install [BlackHole](https://github.com/ExistentialAudio/BlackHole):

```bash
brew install blackhole-2ch
```

1. Open **Audio MIDI Setup** (Applications > Utilities)
2. Click **+** → Create **Multi-Output Device**
3. Check both **BlackHole 2ch** and your **Built-in Output**
4. Set the Multi-Output Device as system output in **System Settings > Sound**
5. Run `livescribe listen` and select BlackHole as the input device

## Troubleshooting

### "espeak-ng is not installed"
```bash
brew install espeak-ng     # macOS
sudo apt install espeak-ng # Linux
```

### "No default input/output device found"
- Check microphone/speaker permissions in System Settings > Privacy & Security
- Use `--list-devices` to find the correct device index

### High latency / "Transcription falling behind"
- Use a faster model: `--model small` or `--model base`
- Enable GPU acceleration: build with `--features metal` (macOS)

### Build errors
- Ensure C/C++ compiler: `xcode-select --install` (macOS) or `sudo apt install build-essential` (Linux)
- cmake is required for whisper-rs: `brew install cmake` (macOS)

## License

MIT
