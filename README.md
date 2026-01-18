# Live Audio Transcription

A simple, effective live transcription tool for macOS that captures audio from your microphone or internal audio and transcribes it in real-time using OpenAI's Whisper model.

## Features

- 🎤 Real-time audio transcription
- 📝 Continuous output to text file with timestamps
- 🔊 Support for both microphone and internal audio (with additional setup)
- ⚙️ Multiple Whisper model sizes (tiny, base, small, medium, large)
- 🎯 Simple CLI interface
- ⏱️ Configurable chunk duration for processing

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PortAudio (required for PyAudio)

On macOS, install via Homebrew:

```bash
brew install portaudio
```

### 3. (Optional) Capture Internal Audio

To transcribe system audio (e.g., from video calls, browser audio), you need to install **BlackHole**:

```bash
brew install blackhole-2ch
```

Then configure your Mac's audio settings:
1. Open **Audio MIDI Setup** (in Applications > Utilities)
2. Click the **+** button and create a **Multi-Output Device**
3. Check both **BlackHole 2ch** and your **Built-in Output**
4. In **System Settings > Sound**, select the Multi-Output Device as your output
5. When running the script, select BlackHole as the input device

## Usage

### Basic Usage (Microphone)

```bash
python live_transcribe.py
```

This will:
- Use your default microphone
- Use the "base" Whisper model
- Save transcription to `transcription.txt`

### Advanced Options

```bash
# Use a specific Whisper model (tiny, base, small, medium, large)
python live_transcribe.py --model small

# Save to a custom file
python live_transcribe.py --output my_notes.txt

# Use a specific audio input device
python live_transcribe.py --device 2

# Adjust chunk duration (seconds of audio to process at once)
python live_transcribe.py --chunk-duration 10

# Combine options
python live_transcribe.py --model medium --output meeting_notes.txt --chunk-duration 8
```

### List Available Audio Devices

Run the script and it will show all available input devices:

```bash
python live_transcribe.py
```

Look for device numbers in the output, then use the `--device` flag to select one.

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

# TODO
Check the main script for more. 

- [ ] Add diarization with name recognition. 
- [ ] Create live annotation cli with shortcuts for notes and questions
- [ ] EasterEgg: this may become part of the simpleassistant.ai stack