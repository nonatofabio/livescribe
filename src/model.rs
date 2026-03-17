use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

/// Known models with their HuggingFace download URLs and approximate sizes.
const MODELS: &[(&str, &str, u64)] = &[
    (
        "tiny",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        75_000_000,
    ),
    (
        "base",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        142_000_000,
    ),
    (
        "small",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        466_000_000,
    ),
    (
        "medium",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        1_533_000_000,
    ),
    (
        "large-v3",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        3_095_000_000,
    ),
    (
        "distil-large-v3",
        "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
        1_528_000_000,
    ),
];

/// Returns the platform-appropriate cache directory for model files.
fn cache_dir() -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("livescribe").join("models")
}

/// Returns the expected file path for a named model.
fn model_path(model_name: &str) -> PathBuf {
    cache_dir().join(format!("ggml-{}.bin", model_name))
}

/// Lists all available model names.
pub fn available_models() -> Vec<&'static str> {
    MODELS.iter().map(|(name, _, _)| *name).collect()
}

/// Ensures a model is present locally, downloading it if necessary.
/// Returns the path to the model file.
pub fn ensure_model(model_name: &str) -> Result<PathBuf> {
    // Check if it's already a file path
    let as_path = Path::new(model_name);
    if as_path.is_file() {
        return Ok(as_path.to_path_buf());
    }

    let path = model_path(model_name);

    if path.exists() {
        println!("Model '{}' found at {}", model_name, path.display());
        return Ok(path);
    }

    let (_, url, expected_size) = MODELS
        .iter()
        .find(|(name, _, _)| *name == model_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown model '{}'. Available models: {}",
                model_name,
                available_models().join(", ")
            )
        })?;

    println!("Downloading model '{}' from HuggingFace...", model_name);

    std::fs::create_dir_all(cache_dir())
        .context("Failed to create model cache directory")?;

    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .context("Failed to create HTTP client")?;

    let mut response = client
        .get(*url)
        .send()
        .context("Failed to start model download")?;

    if !response.status().is_success() {
        bail!(
            "Download failed with HTTP {}: {}",
            response.status(),
            response.status().canonical_reason().unwrap_or("unknown")
        );
    }

    let total_size = response.content_length().unwrap_or(*expected_size);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    // Write to temp file first, then rename for atomicity
    let tmp_path = path.with_extension("bin.tmp");
    let mut file =
        std::fs::File::create(&tmp_path).context("Failed to create temporary model file")?;

    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];

    loop {
        let n = response
            .read(&mut buffer)
            .context("Failed to read download stream")?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buffer[..n])
            .context("Failed to write model data")?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("Download complete");

    std::fs::rename(&tmp_path, &path).context("Failed to finalize model file")?;

    println!("Model saved to {}", path.display());
    Ok(path)
}
