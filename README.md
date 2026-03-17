# Livescribe

A fast, cross-platform live audio transcription tool built in Rust. Captures audio from your microphone and transcribes it in real-time using [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

## Features

- **Non-blocking pipeline**: Recording never stops while transcription runs — audio capture and inference happen on separate threads
- **Overlapping chunks**: Consecutive audio chunks overlap (default 2s) to avoid cutting words at boundaries, with automatic deduplication
- **High-quality default model**: Uses `distil-large-v3` — near large-v3 accuracy at ~6x speed
- **Auto model download**: Models are downloaded from HuggingFace on first run and cached locally
- **Cross-platform**: macOS (CoreAudio), Linux (ALSA), Windows (WASAPI) via cpal
- **GPU acceleration**: Optional Metal (macOS), CUDA (NVIDIA), and CoreML support

## Installation

### Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- A C/C++ compiler (Xcode Command Line Tools on macOS, gcc on Linux, MSVC on Windows)
- On Linux: `libasound2-dev` (Debian/Ubuntu) or `alsa-lib-devel` (Fedora)

### Build

```bash
# Standard CPU build
cargo build --release

# macOS with Metal GPU acceleration
cargo build --release --features metal

# NVIDIA GPU acceleration
cargo build --release --features cuda
```

The binary will be at `target/release/livescribe`.

## Usage

### Basic Usage

```bash
# Start transcribing with defaults (distil-large-v3 model, 8s chunks, 2s overlap)
cargo run --release
```

On first run, the model (~1.5 GB) will be downloaded automatically and cached at `~/.cache/livescribe/models/`.

### List Audio Devices

```bash
cargo run --release -- --list-devices
```

### Options

```bash
livescribe [OPTIONS]

Options:
  -o, --output <FILE>           Output file [default: transcription.txt]
  -m, --model <NAME|PATH>       Whisper model name or path [default: distil-large-v3]
  -d, --device <INDEX>          Audio input device index
  -c, --chunk-duration <SECS>   Audio chunk size in seconds [default: 8]
      --overlap <SECS>          Overlap between chunks in seconds [default: 2]
  -l, --language <CODE>         Language code [default: en]
  -t, --threads <N>             Whisper inference threads
      --list-devices            List audio devices and exit
  -h, --help                    Print help
```

### Examples

```bash
# Transcribe a meeting with medium model
livescribe --model medium --output meeting.txt

# Quick voice notes with faster model
livescribe --model small --chunk-duration 5 --output notes.txt

# Use a specific microphone
livescribe --device 2

# Use a custom model file
livescribe --model /path/to/custom-model.bin
```

## Available Models

Models are auto-downloaded on first use. Choose based on accuracy vs. speed:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 75 MB | Fastest | Good |
| `base` | 142 MB | Fast | Better |
| `small` | 466 MB | Moderate | Great |
| `medium` | 1.5 GB | Slower | Excellent |
| `large-v3` | 3.1 GB | Slowest | Best |
| **`distil-large-v3`** | **1.5 GB** | **Fast** | **Near-best** |

**Recommendation**: The default `distil-large-v3` offers the best accuracy-to-speed ratio. Use `small` for lower resource usage.

## Architecture

Livescribe uses a 3-thread producer-consumer pipeline:

```
[Audio Thread] → [Transcription Thread] → [Output / Main Thread]
    cpal            whisper.cpp               file + stdout
 (never stops)     (CPU-bound)              (dedup + write)
```

1. **Audio thread**: Continuously captures via cpal, assembles overlapping chunks, sends them through a bounded channel
2. **Transcription thread**: Runs whisper.cpp inference on each chunk
3. **Main thread**: Receives results, deduplicates overlap, writes timestamped output

Recording never pauses — if transcription falls behind, the oldest pending chunk is dropped rather than blocking audio capture.

## Output Format

```
============================================================
Transcription started: 2024-01-15 14:30:00
============================================================

[14:30:05] Hello, this is a test of the transcription system.
[14:30:12] It captures audio in chunks and transcribes them.
[14:30:20] The output is saved to a text file with timestamps.

============================================================
Transcription ended: 2024-01-15 14:32:45
============================================================
```

## Capture Internal Audio (macOS)

To transcribe system audio (video calls, browser audio), install [BlackHole](https://github.com/ExistentialAudio/BlackHole):

```bash
brew install blackhole-2ch
```

1. Open **Audio MIDI Setup** (Applications > Utilities)
2. Click **+** → Create **Multi-Output Device**
3. Check both **BlackHole 2ch** and your **Built-in Output**
4. Set the Multi-Output Device as system output in **System Settings > Sound**
5. Run livescribe and select BlackHole as the input device

## Troubleshooting

### "No default input device found"
- Check microphone permissions in System Settings > Privacy & Security > Microphone
- Use `--list-devices` to find the correct device index

### High latency / "Transcription falling behind"
- Use a faster model: `--model small` or `--model base`
- Increase chunk duration: `--chunk-duration 10`
- Enable GPU acceleration: build with `--features metal` (macOS) or `--features cuda`

### Build errors with whisper-rs
- Ensure you have a C/C++ compiler installed
- On macOS: `xcode-select --install`
- On Linux: `sudo apt install build-essential` (Debian/Ubuntu)

## License

MIT
