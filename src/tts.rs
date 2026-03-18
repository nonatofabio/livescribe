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

/// Synthesize all sentences, sending audio chunks through a channel.
pub fn synthesis_loop(
    engine: &mut TtsEngine,
    sentences: Vec<String>,
    total: usize,
    audio_tx: Sender<Vec<f32>>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    check_espeak()?;

    for (i, sentence) in sentences.iter().enumerate() {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        let preview = if sentence.len() > 60 {
            format!("{}...", &sentence[..57])
        } else {
            sentence.clone()
        };
        eprintln!("[{}/{}] {}", i + 1, total, preview);

        match engine.synthesize(sentence) {
            Ok(audio) => {
                if audio.is_empty() {
                    continue;
                }
                if audio_tx.send(audio).is_err() {
                    break;
                }
            }
            Err(e) => {
                eprintln!("[{}/{}] Synthesis failed: {}, skipping", i + 1, total, e);
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
