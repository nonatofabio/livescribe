use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use crossbeam_channel::Sender;

/// Chatterbox model variants.
#[derive(Debug, Clone)]
pub enum ChatterboxModel {
    Turbo,
    Multilingual,
    Original,
}

impl ChatterboxModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Turbo => "turbo",
            Self::Multilingual => "multilingual",
            Self::Original => "original",
        }
    }
}

impl std::str::FromStr for ChatterboxModel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "turbo" => Ok(Self::Turbo),
            "multilingual" => Ok(Self::Multilingual),
            "original" => Ok(Self::Original),
            _ => bail!("Unknown chatterbox model: '{}'. Use: turbo, multilingual, original", s),
        }
    }
}

/// Locate the bundled Python synthesis script.
fn find_script() -> Result<PathBuf> {
    // Check next to the binary
    let exe = std::env::current_exe().context("Failed to get executable path")?;
    let exe_dir = exe.parent().unwrap_or(Path::new("."));

    // Try several locations
    let candidates = [
        exe_dir.join("scripts/chatterbox_synth.py"),
        exe_dir.join("../scripts/chatterbox_synth.py"),
        exe_dir.join("../../scripts/chatterbox_synth.py"),
        PathBuf::from("scripts/chatterbox_synth.py"),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    bail!(
        "Cannot find chatterbox_synth.py. Expected in scripts/ directory.\n\
         Searched: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Check that Python and chatterbox-tts are available.
pub fn check_python() -> Result<()> {
    let output = Command::new("python3")
        .args(["-c", "import chatterbox; print('ok')"])
        .output();

    match output {
        Ok(o) if o.status.success() => Ok(()),
        _ => bail!(
            "Chatterbox TTS requires Python with chatterbox-tts installed:\n  \
             pip install chatterbox-tts\n\n\
             Make sure 'python3' is on your PATH."
        ),
    }
}

/// Manages a Chatterbox Python subprocess for TTS synthesis.
pub struct ChatterboxEngine {
    child: Child,
    sample_rate: u32,
}

impl ChatterboxEngine {
    /// Spawn the Chatterbox Python sidecar process.
    pub fn new(
        model: &ChatterboxModel,
        voice_ref: Option<&Path>,
    ) -> Result<Self> {
        check_python()?;
        let script = find_script()?;

        let mut cmd = Command::new("python3");
        cmd.arg(&script)
            .arg("--model")
            .arg(model.as_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // pass stderr through for progress

        if let Some(ref_path) = voice_ref {
            cmd.arg("--voice").arg(ref_path);
        }

        let mut child = cmd.spawn().context(
            "Failed to start Chatterbox Python process. Is python3 on your PATH?",
        )?;

        // Read sample rate header (4 bytes, little-endian u32)
        let stdout = child.stdout.as_mut().unwrap();
        let mut header = [0u8; 4];
        std::io::Read::read_exact(stdout, &mut header)
            .context("Failed to read sample rate from Chatterbox")?;
        let sample_rate = u32::from_le_bytes(header);

        Ok(Self { child, sample_rate })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Synthesize a single sentence. Sends JSON to stdin, reads framed audio from stdout.
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        let stdin = self.child.stdin.as_mut().unwrap();
        let msg = serde_json::json!({"text": text});
        writeln!(stdin, "{}", msg).context("Failed to write to Chatterbox stdin")?;
        stdin.flush()?;

        // Read framed response: 4 bytes length + N*4 bytes audio
        let stdout = self.child.stdout.as_mut().unwrap();
        let mut len_buf = [0u8; 4];
        std::io::Read::read_exact(stdout, &mut len_buf)
            .context("Failed to read audio length from Chatterbox")?;
        let num_samples = u32::from_le_bytes(len_buf) as usize;

        if num_samples == 0 {
            return Ok(Vec::new());
        }

        let mut audio_buf = vec![0u8; num_samples * 4];
        std::io::Read::read_exact(stdout, &mut audio_buf)
            .context("Failed to read audio data from Chatterbox")?;

        // Convert bytes to f32
        let audio: Vec<f32> = audio_buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(audio)
    }
}

impl Drop for ChatterboxEngine {
    fn drop(&mut self) {
        // Close stdin to signal the Python process to exit
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

/// Synthesize all sentences via Chatterbox, sending audio chunks through a channel.
pub fn synthesis_loop(
    engine: &mut ChatterboxEngine,
    sentences: Vec<String>,
    total: usize,
    audio_tx: Sender<Vec<f32>>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
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
