use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use ort::session::Session;
use ort::value::Tensor;
use regex::Regex;

use crate::voice::PiperConfig;

/// A unit of speech: either a sentence to synthesize or a silence pause.
#[derive(Debug, Clone)]
pub enum SpeechUnit {
    Sentence(String),
    /// Silence duration in seconds.
    Pause(f32),
}

/// Split text into speech units for TTS synthesis.
///
/// Recognizes `[pause]` markers (inserted by the LLM rewriter) and converts
/// them to silence. Splits remaining text on sentence boundaries.
pub fn split_into_speech_units(text: &str) -> Vec<SpeechUnit> {
    let mut units = Vec::new();

    // Split on [pause] markers first
    for segment in text.split("[pause]") {
        let segment = segment.trim();
        if segment.is_empty() {
            if !matches!(units.last(), Some(SpeechUnit::Pause(_))) {
                units.push(SpeechUnit::Pause(SECTION_PAUSE));
            }
            continue;
        }

        // Add section pause before this segment if we already have content
        if !units.is_empty() {
            units.push(SpeechUnit::Pause(SECTION_PAUSE));
        }

        // Split segment into sentences
        let sentences = split_sentences(segment);
        for s in sentences {
            units.push(SpeechUnit::Sentence(s));
        }
    }

    units
}

/// Split text into sentences for individual TTS synthesis.
pub fn split_sentences(text: &str) -> Vec<String> {
    let re = Regex::new(r"(?:[.!?]+[\s]+|[\n]{2,})").unwrap();
    let mut sentences = Vec::new();
    let mut last = 0;

    for mat in re.find_iter(text) {
        let sentence = text[last..mat.start()].trim().to_string();
        if !sentence.is_empty() {
            sentences.push(sentence);
        }
        last = mat.end();
    }

    let remainder = text[last..].trim().to_string();
    if !remainder.is_empty() {
        sentences.push(remainder);
    }

    sentences
}

/// Short breath pause inserted between every sentence (seconds).
const SENTENCE_GAP: f32 = 0.35;
/// Longer pause for explicit [pause] markers between sections (seconds).
const SECTION_PAUSE: f32 = 1.0;

/// Generate silence as f32 samples at the given sample rate.
pub fn generate_silence(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    vec![0.0f32; (duration_secs * sample_rate as f32) as usize]
}

/// Check that espeak-ng is available on the system.
pub fn check_espeak() -> Result<()> {
    match std::process::Command::new("espeak-ng")
        .arg("--version")
        .output()
    {
        Ok(output) if output.status.success() => Ok(()),
        _ => anyhow::bail!(
            "espeak-ng is not installed. Install it:\n  \
             macOS:  brew install espeak-ng\n  \
             Linux:  sudo apt install espeak-ng\n  \
             Windows: download from https://github.com/espeak-ng/espeak-ng/releases"
        ),
    }
}

/// Convert text to IPA phonemes using espeak-ng subprocess.
fn phonemize(text: &str, espeak_voice: &str) -> Result<String> {
    let output = std::process::Command::new("espeak-ng")
        .args(["--ipa", "-q", "-v", espeak_voice, text])
        .output()
        .context("Failed to run espeak-ng")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("espeak-ng failed: {}", stderr);
    }

    let phonemes = String::from_utf8_lossy(&output.stdout)
        .trim()
        .replace('\n', " ")
        .to_string();
    Ok(phonemes)
}

/// Convert IPA phoneme string to Piper phoneme IDs.
fn phonemes_to_ids(phonemes: &str, phoneme_id_map: &HashMap<String, Vec<i64>>) -> Vec<i64> {
    let mut ids = Vec::new();

    // Sentence start + pad
    if let Some(v) = phoneme_id_map.get("^") {
        ids.extend(v);
    }
    if let Some(v) = phoneme_id_map.get("_") {
        ids.extend(v);
    }

    for ch in phonemes.chars() {
        if ch == ' ' {
            if let Some(v) = phoneme_id_map.get(" ") {
                ids.extend(v);
            }
            if let Some(v) = phoneme_id_map.get("_") {
                ids.extend(v);
            }
            continue;
        }

        let key = ch.to_string();
        if let Some(v) = phoneme_id_map.get(&key) {
            ids.extend(v);
            if let Some(pad) = phoneme_id_map.get("_") {
                ids.extend(pad);
            }
        }
    }

    // Sentence end + pad
    if let Some(v) = phoneme_id_map.get("$") {
        ids.extend(v);
    }
    if let Some(v) = phoneme_id_map.get("_") {
        ids.extend(v);
    }

    ids
}

/// Piper TTS engine wrapping an ONNX Runtime session.
pub struct TtsEngine {
    session: Session,
    config: PiperConfig,
}

// Session is not Send by default in ort 2.x RC, but Piper models are safe
// to use from a single thread (we only use it on the synthesis thread).
unsafe impl Send for TtsEngine {}

impl TtsEngine {
    /// Load a Piper ONNX model.
    pub fn new(onnx_path: &Path, config: PiperConfig) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set threads: {}", e))?
            .commit_from_file(onnx_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load Piper ONNX model '{}': {}",
                    onnx_path.display(),
                    e
                )
            })?;

        Ok(Self { session, config })
    }

    /// Output sample rate of this voice.
    pub fn sample_rate(&self) -> u32 {
        self.config.audio.sample_rate
    }

    /// Synthesize a single sentence to f32 audio samples.
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        let phonemes = phonemize(text, &self.config.espeak.voice)?;
        if phonemes.is_empty() {
            return Ok(Vec::new());
        }

        let ids = phonemes_to_ids(&phonemes, &self.config.phoneme_id_map);
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let num_phonemes = ids.len();

        // input: shape [1, num_phonemes], dtype i64
        let input = Tensor::from_array((vec![1i64, num_phonemes as i64], ids))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        // input_lengths: shape [1], dtype i64
        let lengths = Tensor::from_array((vec![1i64], vec![num_phonemes as i64]))
            .map_err(|e| anyhow::anyhow!("Failed to create lengths tensor: {}", e))?;

        // scales: shape [3], dtype f32
        let scales = Tensor::from_array((
            vec![3i64],
            vec![
                self.config.inference.noise_scale,
                self.config.inference.length_scale,
                self.config.inference.noise_w,
            ],
        ))
        .map_err(|e| anyhow::anyhow!("Failed to create scales tensor: {}", e))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input,
                "input_lengths" => lengths,
                "scales" => scales,
            ])
            .map_err(|e| anyhow::anyhow!("Piper inference failed: {}", e))?;

        // output: shape [1, 1, audio_length] — extract the raw f32 slice
        let (_shape, audio_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output tensor: {}", e))?;

        let audio: Vec<f32> = audio_data.to_vec();
        Ok(audio)
    }
}

/// Synthesize speech units (sentences + pauses), sending audio chunks through a channel.
pub fn synthesis_loop(
    engine: &mut TtsEngine,
    units: Vec<SpeechUnit>,
    audio_tx: Sender<Vec<f32>>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    check_espeak()?;

    let total = units.iter().filter(|u| matches!(u, SpeechUnit::Sentence(_))).count();
    let sample_rate = engine.sample_rate();
    let mut sentence_idx = 0;

    for unit in &units {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        match unit {
            SpeechUnit::Pause(duration) => {
                let silence = generate_silence(*duration, sample_rate);
                if audio_tx.send(silence).is_err() {
                    break;
                }
            }
            SpeechUnit::Sentence(sentence) => {
                sentence_idx += 1;
                let preview = if sentence.chars().count() > 60 {
                    let truncated: String = sentence.chars().take(57).collect();
                    format!("{}...", truncated)
                } else {
                    sentence.clone()
                };
                eprintln!("[{}/{}] {}", sentence_idx, total, preview);

                match engine.synthesize(sentence) {
                    Ok(audio) => {
                        if audio.is_empty() {
                            continue;
                        }
                        if audio_tx.send(audio).is_err() {
                            break;
                        }
                        // Breath pause between sentences
                        let gap = generate_silence(SENTENCE_GAP, sample_rate);
                        if audio_tx.send(gap).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("[{}/{}] Synthesis failed: {}, skipping", sentence_idx, total, e);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. This is a test. How are you?";
        let sentences = split_sentences(text);
        assert_eq!(
            sentences,
            vec!["Hello world", "This is a test", "How are you?"]
        );
    }

    #[test]
    fn test_split_sentences_newlines() {
        let text = "First paragraph.\n\nSecond paragraph.";
        let sentences = split_sentences(text);
        assert_eq!(sentences, vec!["First paragraph", "Second paragraph."]);
    }

    #[test]
    fn test_split_sentences_empty() {
        let sentences = split_sentences("");
        assert!(sentences.is_empty());
    }
}
