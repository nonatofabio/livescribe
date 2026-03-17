mod audio;
mod dedup;
mod model;
mod output;
mod transcribe;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Result};
use clap::Parser;

pub use transcribe::{Segment, TranscriptionResult};

/// Audio chunk passed from the capture thread to the transcription thread.
pub struct AudioChunk {
    /// PCM samples as f32, mono, 16kHz.
    pub samples: Vec<f32>,
    /// Wall-clock time when this chunk's recording started.
    pub timestamp: chrono::DateTime<chrono::Local>,
    /// Sequential chunk index (0, 1, 2, ...).
    pub chunk_index: u64,
    /// Number of samples at the start that overlap with the previous chunk.
    pub overlap_samples: usize,
}

#[derive(Parser, Debug)]
#[command(name = "livescribe", about = "Live audio transcription using Whisper")]
struct Cli {
    /// Output file for transcription.
    #[arg(short, long, default_value = "transcription.txt")]
    output: PathBuf,

    /// Whisper model name or path to a .bin file.
    #[arg(short, long, default_value = "distil-large-v3")]
    model: String,

    /// Audio input device index (use --list-devices to see available).
    #[arg(short, long)]
    device: Option<usize>,

    /// Duration of each audio chunk in seconds.
    #[arg(short, long, default_value_t = 8)]
    chunk_duration: u32,

    /// Overlap between consecutive chunks in seconds.
    #[arg(long, default_value_t = 2)]
    overlap: u32,

    /// Language code for transcription.
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Number of threads for Whisper inference.
    #[arg(short = 't', long)]
    threads: Option<u32>,

    /// List available audio devices and exit.
    #[arg(long)]
    list_devices: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_devices {
        return audio::list_devices();
    }

    if cli.overlap >= cli.chunk_duration {
        bail!(
            "Overlap ({}s) must be less than chunk duration ({}s)",
            cli.overlap,
            cli.chunk_duration
        );
    }

    // Resolve model path (downloads if needed)
    let model_path = if PathBuf::from(&cli.model).is_file() {
        PathBuf::from(&cli.model)
    } else {
        model::ensure_model(&cli.model)?
    };

    // Load whisper model
    println!("Loading Whisper model '{}'...", cli.model);
    let ctx = transcribe::load_context(&model_path)?;
    println!("Model loaded.");

    // Resolve audio device
    let device = audio::get_device(cli.device)?;
    let device_name = audio::device_name(&device);
    println!("Using audio device: {}", device_name);

    // Shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let s = shutdown.clone();
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            s.store(true, Ordering::SeqCst);
        })?;
    }

    // Channels
    let (audio_tx, audio_rx) = crossbeam_channel::bounded::<AudioChunk>(2);
    let (result_tx, result_rx) = crossbeam_channel::bounded::<TranscriptionResult>(4);

    let overlap_ratio = cli.overlap as f64 / cli.chunk_duration as f64;

    println!("\n{}", "=".repeat(60));
    println!("Recording and transcribing... Press Ctrl+C to stop");
    println!(
        "  Chunk: {}s | Overlap: {}s | Model: {}",
        cli.chunk_duration, cli.overlap, cli.model
    );
    println!("{}\n", "=".repeat(60));

    // Spawn audio capture thread
    let shutdown_a = shutdown.clone();
    let chunk_dur = cli.chunk_duration;
    let overlap_secs = cli.overlap;
    let audio_handle = std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            if let Err(e) =
                audio::capture_loop(device, chunk_dur, overlap_secs, audio_tx, shutdown_a)
            {
                eprintln!("Audio capture error: {}", e);
            }
        })?;

    // Spawn transcription thread
    let shutdown_t = shutdown.clone();
    let language = cli.language.clone();
    let n_threads = cli.threads;
    let transcribe_handle = std::thread::Builder::new()
        .name("transcription".into())
        .spawn(move || {
            if let Err(e) = transcribe::transcription_loop(
                &ctx,
                audio_rx,
                result_tx,
                &language,
                n_threads,
                shutdown_t,
            ) {
                eprintln!("Transcription error: {}", e);
            }
        })?;

    // Output loop on main thread
    output::output_loop(result_rx, &cli.output, overlap_ratio)?;

    // Join workers
    audio_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Audio thread panicked"))?;
    transcribe_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Transcription thread panicked"))?;

    println!("\nTranscription saved to: {}", cli.output.display());
    Ok(())
}
