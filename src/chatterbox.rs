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
            _ => bail!(
                "Unknown chatterbox model: '{}'. Use: turbo, multilingual, original",
                s
            ),
        }
    }
}

/// The Python synthesis script, embedded in the binary at compile time.
const SYNTH_SCRIPT: &str = include_str!("../scripts/chatterbox_synth.py");

// ---------------------------------------------------------------------------
// Managed venv in ~/.cache/livescribe/chatterbox-venv/
// ---------------------------------------------------------------------------

/// Returns the path to the managed Chatterbox venv.
fn venv_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("livescribe")
        .join("chatterbox-venv")
}

/// Returns the python3 binary inside the managed venv.
fn venv_python() -> PathBuf {
    let venv = venv_dir();
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python3")
    }
}

/// Marker file that records a successful chatterbox-tts install.
fn installed_marker() -> PathBuf {
    venv_dir().join(".chatterbox-installed")
}

/// Find a compatible system python (3.11 or 3.12) for the Chatterbox venv.
///
/// Chatterbox pins old numpy (<1.26) which doesn't build on Python 3.13+.
/// We try specific versioned binaries first to avoid the system default
/// if it's too new.
fn find_system_python() -> Result<String> {
    // Prefer 3.11/3.12 — known compatible with chatterbox-tts
    let candidates = [
        "python3.12",
        "python3.11",
        "python3.13",
        "python3",
        "python",
    ];

    for candidate in candidates {
        if let Ok(output) = Command::new(candidate).args(["--version"]).output() {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout);
                let version = version.trim();

                // Parse major.minor
                if let Some(ver_str) = version.strip_prefix("Python ") {
                    let parts: Vec<&str> = ver_str.split('.').collect();
                    if let (Some(major), Some(minor)) = (
                        parts.first().and_then(|s| s.parse::<u32>().ok()),
                        parts.get(1).and_then(|s| s.parse::<u32>().ok()),
                    ) {
                        if major == 3 && (11..=12).contains(&minor) {
                            eprintln!("Using {} for Chatterbox venv", version);
                            return Ok(candidate.to_string());
                        }
                        // Skip 3.14+ — incompatible with chatterbox's numpy pin
                        if major == 3 && minor >= 14 {
                            continue;
                        }
                        // 3.13 — might work, use as fallback
                        if major == 3 && minor == 13 {
                            eprintln!(
                                "Warning: {} may have issues with chatterbox-tts. \
                                 Python 3.11 or 3.12 recommended.",
                                version
                            );
                            return Ok(candidate.to_string());
                        }
                    }
                }
            }
        }
    }

    bail!(
        "No compatible Python found for Chatterbox (needs 3.11 or 3.12).\n\
         Your system Python 3.14 is too new for chatterbox-tts.\n\n\
         Install a compatible version:\n  \
         macOS:  brew install python@3.12\n  \
         Linux:  sudo apt install python3.12 python3.12-venv"
    )
}

/// Ensures the managed venv exists with chatterbox-tts installed.
/// Creates it on first use. This is the only time pip runs.
fn ensure_venv() -> Result<PathBuf> {
    let python = venv_python();

    // Fast path: venv already set up
    if python.exists() && installed_marker().exists() {
        return Ok(python);
    }

    let venv = venv_dir();
    let sys_python = find_system_python()?;

    // Step 1: Create venv
    if !python.exists() {
        println!("Creating Chatterbox Python environment at {}...", venv.display());
        let status = Command::new(&sys_python)
            .args(["-m", "venv", &venv.to_string_lossy()])
            .status()
            .context("Failed to create Python venv")?;

        if !status.success() {
            bail!(
                "Failed to create venv. You may need python3-venv:\n  \
                 sudo apt install python3-venv"
            );
        }
    }

    // Step 2: Upgrade pip and setuptools (fresh venvs ship outdated ones)
    if !installed_marker().exists() {
        println!("Upgrading pip and setuptools...");
        // Pin setuptools<78: newer versions removed pkg_resources.ImpImporter
        // which chatterbox's pinned numpy<1.26 needs to build from source.
        let status = Command::new(&python)
            .args([
                "-m", "pip", "install", "--upgrade",
                "pip", "setuptools<78", "wheel",
            ])
            .status()
            .context("Failed to upgrade pip/setuptools")?;

        if !status.success() {
            eprintln!("Warning: pip upgrade failed, continuing anyway...");
        }

        // Step 3: Install chatterbox-tts
        println!("Installing chatterbox-tts (this may take a few minutes on first run)...");

        let status = Command::new(&python)
            .args(["-m", "pip", "install", "chatterbox-tts"])
            .status()
            .context("Failed to install chatterbox-tts")?;

        if !status.success() {
            bail!(
                "pip install chatterbox-tts failed.\n\
                 Try manually: {} -m pip install chatterbox-tts",
                python.display()
            );
        }

        // Write marker so we don't re-install next time
        std::fs::write(installed_marker(), "ok").ok();
        println!("Chatterbox TTS installed successfully.");
    }

    Ok(python)
}

/// Writes the embedded Python script to the cache directory.
fn ensure_script() -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("livescribe");
    std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;

    let script_path = cache_dir.join("chatterbox_synth.py");
    std::fs::write(&script_path, SYNTH_SCRIPT)
        .context("Failed to write chatterbox_synth.py to cache")?;

    Ok(script_path)
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Manages a Chatterbox Python subprocess for TTS synthesis.
pub struct ChatterboxEngine {
    child: Child,
    sample_rate: u32,
}

impl ChatterboxEngine {
    /// Spawn the Chatterbox Python sidecar process.
    ///
    /// On first use, this will:
    /// 1. Create an isolated venv at ~/.cache/livescribe/chatterbox-venv/
    /// 2. pip install chatterbox-tts into it
    /// 3. Run the synthesis script using that venv's python
    ///
    /// Subsequent runs reuse the venv instantly.
    pub fn new(model: &ChatterboxModel, voice_ref: Option<&Path>) -> Result<Self> {
        let python = ensure_venv()?;
        let script = ensure_script()?;

        let mut cmd = Command::new(&python);
        cmd.arg(&script)
            .arg("--model")
            .arg(model.as_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        if let Some(ref_path) = voice_ref {
            cmd.arg("--voice").arg(ref_path);
        }

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "Failed to start Chatterbox. Venv python: {}",
                python.display()
            )
        })?;

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

        let audio: Vec<f32> = audio_buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(audio)
    }
}

impl Drop for ChatterboxEngine {
    fn drop(&mut self) {
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
