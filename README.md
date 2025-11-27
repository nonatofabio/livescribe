# LiveScribe

A modern, real-time audio transcription app for macOS using OpenAI's Whisper model. Now with a beautiful desktop GUI built with Tauri!

## Features

- 🎤 Real-time audio transcription with Whisper
- 🖥️ Modern desktop app (Tauri - lightweight, ~15MB)
- 📝 Live transcript with timestamps
- 🔊 Support for microphone and internal audio (via BlackHole)
- ⚙️ Multiple Whisper model sizes (tiny → large)
- 🌍 Multi-language support
- 📋 Copy to clipboard / Save to file
- ⏱️ Configurable chunk duration
- 🎨 Beautiful dark theme UI

## Installation

### Prerequisites

1. **Rust** (for Tauri)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. **Node.js** (v18+)
```bash
brew install node
```

3. **pyenv** (Python version manager)
```bash
brew install pyenv

# Add to your shell (~/.zshrc or ~/.bashrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc

# Restart shell or run
source ~/.zshrc
```

4. **PortAudio** (required for PyAudio)
```bash
brew install portaudio
```

### Setup

```bash
# Clone the repo
cd livescribe

# Install Python version (uses .python-version file)
pyenv install 3.11.9
pyenv local 3.11.9  # Already set via .python-version

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python backend dependencies
pip install -r backend/requirements.txt

# Install Node dependencies
npm install
```

### (Optional) Capture Internal Audio

To transcribe system audio (video calls, browser audio), install **BlackHole**:

```bash
brew install blackhole-2ch
```

Configure your Mac's audio:
1. Open **Audio MIDI Setup** (Applications > Utilities)
2. Click **+** → Create **Multi-Output Device**
3. Check **BlackHole 2ch** and **Built-in Output**
4. In **System Settings > Sound**, select Multi-Output as output
5. In LiveScribe, select BlackHole as input device

## Usage

### Desktop App (Recommended)

**Development mode:**
```bash
# Activate virtual environment first
source .venv/bin/activate

# Terminal 1: Start Python backend
python backend/transcription_server.py

# Terminal 2: Start Tauri dev
npm run tauri:dev
```

Or use the convenience script:
```bash
chmod +x scripts/dev.sh
./scripts/dev.sh
```

**Build for production:**
```bash
npm run tauri:build
```

The built app will be in `src-tauri/target/release/bundle/`.

### CLI Mode (Original)

Still works for headless/scripting use:

```bash
python live_transcribe.py

# Options
python live_transcribe.py --model small --output notes.txt --chunk-duration 10
```

## Whisper Model Sizes

Choose based on accuracy vs. speed trade-off:

| Model  | Size  | Speed      | Accuracy |
|--------|-------|------------|----------|
| tiny   | 39 MB | Fastest    | Good     |
| base   | 74 MB | Fast       | Better   |
| small  | 244 MB| Moderate   | Great    |
| medium | 769 MB| Slower     | Excellent|
| large  | 1550 MB| Slowest   | Best     |

**Recommendation**: Start with `base` for a good balance. Use `small` or `medium` for better accuracy.

## Examples

### Transcribe a Meeting

```bash
python live_transcribe.py --model small --output meeting_2024.txt
```

### Quick Voice Notes

```bash
python live_transcribe.py --model tiny --chunk-duration 3 --output quick_notes.txt
```

### High-Quality Lecture Recording

```bash
python live_transcribe.py --model medium --chunk-duration 10 --output lecture.txt
```

## Output Format

The transcription file includes timestamps for each chunk:

```
============================================================
Transcription started: 2024-01-15 14:30:00
============================================================

[14:30:05] Hello, this is a test of the transcription system.
[14:30:12] It captures audio in chunks and transcribes them using Whisper.
[14:30:20] The output is saved to a text file with timestamps.

============================================================
Transcription ended: 2024-01-15 14:32:45
============================================================
```

## Stopping the Script

Press `Ctrl+C` to stop recording and transcription. The output will be properly saved and closed.

## Troubleshooting

### "No module named 'pyaudio'"

Install PortAudio first:
```bash
brew install portaudio
pip install pyaudio
```

### "Cannot open audio device"

- Check your microphone permissions in **System Settings > Privacy & Security > Microphone**
- Try listing devices with the script to find the correct device index
- Make sure no other application is using the microphone

### Poor Transcription Quality

- Use a larger Whisper model (e.g., `--model medium`)
- Increase chunk duration (`--chunk-duration 10`)
- Ensure you're speaking clearly and close to the microphone
- Reduce background noise

### High CPU Usage

- Use a smaller model (e.g., `--model tiny` or `--model base`)
- Increase chunk duration to process less frequently

## Tips

1. **For best results**: Use the `small` or `medium` model
2. **For real-time speed**: Use `tiny` or `base` model with shorter chunks
3. **For meetings**: Use 8-10 second chunks with `small` model
4. **For voice notes**: Use 3-5 second chunks with `base` model

## License

This project uses OpenAI's Whisper model, which is released under the MIT License.