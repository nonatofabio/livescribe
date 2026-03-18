use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// Base URL for Piper voice models on HuggingFace.
const HF_BASE_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main";

/// Known Piper voices: (name, hf_relative_path, approximate .onnx size in bytes).
const VOICES: &[(&str, &str, u64)] = &[
    (
        "en_US-amy-medium",
        "en/en_US/amy/medium/en_US-amy-medium",
        63_200_000,
    ),
    (
        "en_US-lessac-medium",
        "en/en_US/lessac/medium/en_US-lessac-medium",
        63_200_000,
    ),
    (
        "en_US-ryan-medium",
        "en/en_US/ryan/medium/en_US-ryan-medium",
        63_200_000,
    ),
    (
        "en_US-joe-medium",
        "en/en_US/joe/medium/en_US-joe-medium",
        63_200_000,
    ),
    (
        "en_US-lessac-high",
        "en/en_US/lessac/high/en_US-lessac-high",
        75_000_000,
    ),
    (
        "en_GB-alba-medium",
        "en/en_GB/alba/medium/en_GB-alba-medium",
        63_200_000,
    ),
    (
        "en_GB-jenny_dioco-medium",
        "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium",
        63_200_000,
    ),
];

/// Piper model config (deserialized from .onnx.json).
#[derive(Debug, Deserialize)]
pub struct PiperConfig {
    pub audio: AudioConfig,
    pub espeak: EspeakConfig,
    pub inference: InferenceConfig,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
}

#[derive(Debug, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
}

#[derive(Debug, Deserialize)]
pub struct EspeakConfig {
    pub voice: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

/// Cache directory for voice model files.
fn voice_cache_dir() -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("livescribe").join("voices")
}

/// Lists all available voice names.
pub fn list_voices() {
    println!("Available Piper voices:");
    for (name, _, size) in VOICES {
        println!("  {} (~{}MB)", name, size / 1_000_000);
    }
    println!("\nUse: livescribe speak <FILE> --voice <NAME>");
}

/// Ensures a voice model is present locally, downloading if needed.
/// Returns the path to the .onnx file and the parsed config.
pub fn ensure_voice(voice_name: &str) -> Result<(PathBuf, PiperConfig)> {
    // Check if it's already a direct file path
    let as_path = Path::new(voice_name);
    if as_path.is_file() && voice_name.ends_with(".onnx") {
        let json_path = PathBuf::from(format!("{}.json", voice_name));
        let config = load_config(&json_path)?;
        return Ok((as_path.to_path_buf(), config));
    }

    let onnx_path = voice_cache_dir().join(format!("{}.onnx", voice_name));
    let json_path = voice_cache_dir().join(format!("{}.onnx.json", voice_name));

    if onnx_path.exists() && json_path.exists() {
        println!("Voice '{}' found at {}", voice_name, onnx_path.display());
        let config = load_config(&json_path)?;
        return Ok((onnx_path, config));
    }

    let (_, hf_path, expected_size) = VOICES
        .iter()
        .find(|(name, _, _)| *name == voice_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown voice '{}'. Available: {}",
                voice_name,
                VOICES
                    .iter()
                    .map(|(n, _, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    std::fs::create_dir_all(voice_cache_dir()).context("Failed to create voice cache directory")?;

    // Download config JSON first (small)
    let json_url = format!("{}/{}.onnx.json", HF_BASE_URL, hf_path);
    println!("Downloading voice config...");
    download_file(&json_url, &json_path, None)?;

    // Download ONNX model (large, with progress bar)
    let onnx_url = format!("{}/{}.onnx", HF_BASE_URL, hf_path);
    println!("Downloading voice model '{}'...", voice_name);
    download_file(&onnx_url, &onnx_path, Some(*expected_size))?;

    let config = load_config(&json_path)?;
    println!("Voice saved to {}", onnx_path.display());
    Ok((onnx_path, config))
}

fn load_config(json_path: &Path) -> Result<PiperConfig> {
    let file = std::fs::File::open(json_path)
        .with_context(|| format!("Failed to open config: {}", json_path.display()))?;
    serde_json::from_reader(file).context("Failed to parse Piper config JSON")
}

fn download_file(url: &str, dest: &Path, expected_size: Option<u64>) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .context("Failed to create HTTP client")?;

    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to download: {}", url))?;

    if !response.status().is_success() {
        bail!("Download failed with HTTP {}: {}", response.status(), url);
    }

    let total_size = response.content_length().or(expected_size);

    let tmp_path = dest.with_extension("tmp");
    let mut file =
        std::fs::File::create(&tmp_path).context("Failed to create temporary file")?;

    if let Some(size) = total_size {
        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                )?
                .progress_chars("#>-"),
        );

        let mut downloaded: u64 = 0;
        let mut buffer = [0u8; 8192];
        loop {
            let n = response
                .read(&mut buffer)
                .context("Failed to read download stream")?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut file, &buffer[..n])?;
            downloaded += n as u64;
            pb.set_position(downloaded);
        }
        pb.finish_with_message("Done");
    } else {
        std::io::copy(&mut response, &mut file).context("Failed to download file")?;
    }

    std::fs::rename(&tmp_path, dest).context("Failed to finalize downloaded file")?;
    Ok(())
}
