use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;

use crate::audio::WHISPER_SAMPLE_RATE;
use crate::dedup;
use crate::TranscriptionResult;

/// Receives transcription results and writes them to a file and stdout.
///
/// Runs on the main thread. Applies overlap deduplication so consecutive
/// chunks produce clean, non-duplicated output.
pub fn output_loop(
    result_rx: Receiver<TranscriptionResult>,
    output_path: &Path,
    overlap_ratio: f64,
) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .with_context(|| format!("Failed to open output file: {}", output_path.display()))?;

    // Session header
    let header = format!(
        "\n\n{sep}\nTranscription started: {ts}\n{sep}\n\n",
        sep = "=".repeat(60),
        ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
    );
    file.write_all(header.as_bytes())?;
    print!("{}", header);

    let mut prev_text = String::new();

    while let Ok(result) = result_rx.recv() {
        if result.raw_text.is_empty() {
            eprintln!("[Chunk {}] No speech detected", result.chunk_index);
            continue;
        }

        // Layer 1: Filter segments that fall within the overlap region
        let deduped = if result.chunk_index == 0 || prev_text.is_empty() {
            result.raw_text.clone()
        } else {
            let filtered =
                dedup::filter_overlap_segments(&result.segments, result.overlap_samples, WHISPER_SAMPLE_RATE);

            if filtered.is_empty() {
                // All segments were in the overlap — fall back to word-level dedup
                dedup::deduplicate(&prev_text, &result.raw_text, overlap_ratio)
            } else {
                let filtered_text: String = filtered
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join("")
                    .trim()
                    .to_string();

                // Layer 2: Word-level dedup as safety net
                dedup::deduplicate(&prev_text, &filtered_text, overlap_ratio * 0.5)
            }
        };

        if !deduped.is_empty() {
            let timestamp = result.timestamp.format("%H:%M:%S");
            let line = format!("[{}] {}\n", timestamp, deduped);

            print!("{}", line);
            file.write_all(line.as_bytes())?;
            file.flush()?;
        }

        prev_text = result.raw_text;
    }

    // Session footer
    let footer = format!(
        "\n{sep}\nTranscription ended: {ts}\n{sep}\n",
        sep = "=".repeat(60),
        ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
    );
    file.write_all(footer.as_bytes())?;
    print!("{}", footer);

    Ok(())
}
