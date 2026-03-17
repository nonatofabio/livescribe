use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::AudioChunk;

/// A single transcribed segment with timing info relative to its chunk.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in centiseconds from the chunk start.
    pub start_cs: i64,
    /// End time in centiseconds from the chunk start.
    pub end_cs: i64,
    /// The transcribed text of this segment.
    pub text: String,
}

/// Result of transcribing a single audio chunk.
pub struct TranscriptionResult {
    pub chunk_index: u64,
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub raw_text: String,
    pub segments: Vec<Segment>,
    pub overlap_samples: usize,
}

/// Loads a whisper.cpp model and returns the context.
pub fn load_context(model_path: &Path) -> Result<WhisperContext> {
    let params = WhisperContextParameters::default();
    WhisperContext::new_with_params(model_path.to_str().unwrap_or(""), params)
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {:?}", e))
}

/// Receives audio chunks and runs whisper inference, sending results downstream.
///
/// Runs on a dedicated thread. Processes chunks sequentially — while one chunk
/// is being transcribed, the audio capture thread continues recording the next.
pub fn transcription_loop(
    ctx: &WhisperContext,
    audio_rx: Receiver<AudioChunk>,
    result_tx: Sender<TranscriptionResult>,
    language: &str,
    n_threads: Option<u32>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {:?}", e))?;

    while let Ok(chunk) = audio_rx.recv() {
        if shutdown.load(Ordering::SeqCst) && audio_rx.is_empty() {
            break;
        }

        eprintln!("[Chunk {}] Transcribing...", chunk.chunk_index);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_language(Some(language));
        params.set_temperature(0.0);
        params.set_no_speech_thold(0.6);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);

        if let Some(t) = n_threads {
            params.set_n_threads(t as i32);
        }

        // Run inference — this is the CPU-intensive part
        match state.full(params, &chunk.samples) {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "[Chunk {}] Transcription failed: {:?}, skipping",
                    chunk.chunk_index, e
                );
                continue;
            }
        }

        // Extract segments
        let n_segments = match state.full_n_segments() {
            Ok(n) => n,
            Err(e) => {
                eprintln!(
                    "[Chunk {}] Failed to get segment count: {:?}",
                    chunk.chunk_index, e
                );
                continue;
            }
        };

        let mut segments = Vec::new();
        for i in 0..n_segments {
            let text = match state.full_get_segment_text(i) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[Chunk {}] Failed to get segment {} text: {:?}", chunk.chunk_index, i, e);
                    continue;
                }
            };
            let t0 = state
                .full_get_segment_t0(i)
                .unwrap_or(0);
            let t1 = state
                .full_get_segment_t1(i)
                .unwrap_or(0);

            segments.push(Segment {
                start_cs: t0,
                end_cs: t1,
                text,
            });
        }

        let raw_text: String = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string();

        let result = TranscriptionResult {
            chunk_index: chunk.chunk_index,
            timestamp: chunk.timestamp,
            raw_text,
            segments,
            overlap_samples: chunk.overlap_samples,
        };

        if result_tx.send(result).is_err() {
            break;
        }
    }

    Ok(())
}
