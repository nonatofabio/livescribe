mod audio;
mod chatterbox;
mod dedup;
mod document;
mod model;
mod output;
mod rewrite;
mod transcribe;
mod tts;
mod voice;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

pub use transcribe::{Segment, TranscriptionResult};

/// Audio chunk passed from the capture thread to the transcription thread.
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub chunk_index: u64,
    pub overlap_samples: usize,
}

#[derive(Parser, Debug)]
#[command(
    name = "livescribe",
    about = "Live audio transcription and text-to-speech",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Transcribe live audio from microphone (real-time STT).
    Listen(ListenArgs),
    /// Read a document aloud (text-to-speech).
    Speak(SpeakArgs),
}

#[derive(clap::Args, Debug)]
struct ListenArgs {
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
    /// List available audio input devices and exit.
    #[arg(long)]
    list_devices: bool,
}

#[derive(clap::Args, Debug)]
struct SpeakArgs {
    /// Document file to read aloud (.txt, .md, .pdf).
    file: PathBuf,
    /// TTS engine: "piper" (fast, local) or "chatterbox" (high quality, needs Python).
    #[arg(short, long, default_value = "piper")]
    engine: String,
    /// Voice: Piper voice name, or path to reference .wav for Chatterbox voice cloning.
    #[arg(short, long, default_value = "en_US-amy-medium")]
    voice: String,
    /// Save audio to WAV file.
    #[arg(short, long)]
    save: Option<PathBuf>,
    /// Speech speed multiplier (0.5 = half speed, 2.0 = double). Piper only.
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    /// Audio output device index.
    #[arg(short, long)]
    device: Option<usize>,
    /// List available voices and exit.
    #[arg(long)]
    list_voices: bool,
    /// List available audio output devices and exit.
    #[arg(long)]
    list_devices: bool,
    /// Don't play audio (only useful with --save).
    #[arg(long)]
    no_play: bool,
    /// Rewrite document with an LLM for natural TTS narration.
    /// Replaces diagrams with descriptions, adds pauses, smooths formatting.
    /// Requires AWS credentials for Bedrock access.
    #[arg(long)]
    rewrite: bool,
    /// LLM model ID for rewriting (Bedrock model ID).
    #[arg(long, default_value = "us.anthropic.claude-sonnet-4-6")]
    rewrite_model: String,
    /// Save the LLM-rewritten text to a file for inspection.
    #[arg(long)]
    save_rewrite: Option<PathBuf>,
    /// Save the extracted document text (pre-rewrite) to a file for inspection.
    #[arg(long)]
    save_extract: Option<PathBuf>,
    /// Show verbose debug output (API calls, chunk sizes, timing).
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Listen(args) => run_listen(args),
        Command::Speak(args) => run_speak(args),
    }
}

// ---------------------------------------------------------------------------
// Listen (STT) — existing functionality, moved from old main()
// ---------------------------------------------------------------------------

fn run_listen(args: ListenArgs) -> Result<()> {
    if args.list_devices {
        return audio::list_devices();
    }

    if args.overlap >= args.chunk_duration {
        bail!(
            "Overlap ({}s) must be less than chunk duration ({}s)",
            args.overlap,
            args.chunk_duration
        );
    }

    let model_path = if PathBuf::from(&args.model).is_file() {
        PathBuf::from(&args.model)
    } else {
        model::ensure_model(&args.model)?
    };

    println!("Loading Whisper model '{}'...", args.model);
    let ctx = transcribe::load_context(&model_path)?;
    println!("Model loaded.");

    let device = audio::get_device(args.device)?;
    let device_name = audio::device_name(&device);
    println!("Using audio device: {}", device_name);

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let s = shutdown.clone();
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            s.store(true, Ordering::SeqCst);
        })?;
    }

    let (audio_tx, audio_rx) = crossbeam_channel::bounded::<AudioChunk>(2);
    let (result_tx, result_rx) = crossbeam_channel::bounded::<TranscriptionResult>(4);

    let overlap_ratio = args.overlap as f64 / args.chunk_duration as f64;

    println!("\n{}", "=".repeat(60));
    println!("Recording and transcribing... Press Ctrl+C to stop");
    println!(
        "  Chunk: {}s | Overlap: {}s | Model: {}",
        args.chunk_duration, args.overlap, args.model
    );
    println!("{}\n", "=".repeat(60));

    let shutdown_a = shutdown.clone();
    let chunk_dur = args.chunk_duration;
    let overlap_secs = args.overlap;
    let audio_handle = std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            if let Err(e) =
                audio::capture_loop(device, chunk_dur, overlap_secs, audio_tx, shutdown_a)
            {
                eprintln!("Audio capture error: {}", e);
            }
        })?;

    let shutdown_t = shutdown.clone();
    let language = args.language.clone();
    let n_threads = args.threads;
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

    output::output_loop(result_rx, &args.output, overlap_ratio)?;

    audio_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Audio thread panicked"))?;
    transcribe_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Transcription thread panicked"))?;

    println!("\nTranscription saved to: {}", args.output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Speak (TTS)
// ---------------------------------------------------------------------------

fn run_speak(args: SpeakArgs) -> Result<()> {
    if args.list_voices {
        voice::list_voices();
        return Ok(());
    }
    if args.list_devices {
        return audio::list_output_devices();
    }

    if args.no_play && args.save.is_none() {
        bail!("--no-play requires --save to specify an output file");
    }

    // 1. Parse document
    println!("Reading {}...", args.file.display());
    let mut text = document::extract_text(&args.file)?;
    if text.trim().is_empty() {
        bail!("No text found in {}", args.file.display());
    }

    // 1a. Optionally save the extracted text for inspection.
    if let Some(ref path) = args.save_extract {
        std::fs::write(path, &text)
            .with_context(|| format!("Failed to save extract to {}", path.display()))?;
        println!("Extracted text saved to {}", path.display());
    }

    // 1b. Optional LLM rewrite for natural narration
    if args.rewrite {
        println!("Rewriting for natural speech (model: {})...", args.rewrite_model);
        text = rewrite::rewrite_for_speech(
            &text,
            Some(&args.rewrite_model),
            args.verbose,
            args.save_rewrite.as_deref(),
        )?;
        println!("Rewrite complete ({} chars).", text.len());
    }

    let units = tts::split_into_speech_units(&text);
    let sentence_count = units
        .iter()
        .filter(|u| matches!(u, tts::SpeechUnit::Sentence(_)))
        .count();
    println!("Found {} sentences to speak.", sentence_count);

    // 2. Shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let s = shutdown.clone();
        ctrlc::set_handler(move || {
            eprintln!("\nStopping...");
            s.store(true, Ordering::SeqCst);
        })?;
    }

    // 3. Channel: synthesis -> playback/save
    let (audio_tx, audio_rx) = crossbeam_channel::bounded::<Vec<f32>>(4);

    // 4. Route to the right engine
    let source_rate: u32;

    let synth_handle = match args.engine.as_str() {
        "piper" => {
            let (onnx_path, mut config) = voice::ensure_voice(&args.voice)?;
            config.inference.length_scale /= args.speed;

            println!("Loading Piper TTS (voice: {})...", args.voice);
            let mut engine = tts::TtsEngine::new(&onnx_path, config)?;
            source_rate = engine.sample_rate();
            println!("Engine loaded. Sample rate: {}Hz", source_rate);

            let shutdown_s = shutdown.clone();
            std::thread::Builder::new()
                .name("tts-synthesis".into())
                .spawn(move || {
                    if let Err(e) =
                        tts::synthesis_loop(&mut engine, units, audio_tx, shutdown_s)
                    {
                        eprintln!("Synthesis error: {}", e);
                    }
                })?
        }
        "chatterbox" => {
            let voice_ref = if args.voice != "en_US-amy-medium"
                && std::path::Path::new(&args.voice).exists()
            {
                Some(PathBuf::from(&args.voice))
            } else {
                None
            };

            println!(
                "Loading Chatterbox TTS{}...",
                voice_ref
                    .as_ref()
                    .map(|p| format!(" (voice: {})", p.display()))
                    .unwrap_or_default()
            );
            let mut engine =
                chatterbox::ChatterboxEngine::new(voice_ref.as_deref())?;
            source_rate = engine.sample_rate();
            println!("Engine loaded. Sample rate: {}Hz", source_rate);

            let shutdown_s = shutdown.clone();
            std::thread::Builder::new()
                .name("chatterbox-synthesis".into())
                .spawn(move || {
                    if let Err(e) = chatterbox::synthesis_loop(
                        &mut engine,
                        units,
                        audio_tx,
                        shutdown_s,
                    ) {
                        eprintln!("Synthesis error: {}", e);
                    }
                })?
        }
        other => bail!(
            "Unknown engine '{}'. Available: piper, chatterbox",
            other
        ),
    };

    // 5. Playback and/or save
    let want_play = !args.no_play;
    let want_save = args.save.is_some();

    if want_play && want_save {
        let save_path = args.save.unwrap();
        let device = audio::get_output_device(args.device)?;
        let shutdown_p = shutdown.clone();

        let (play_tx, play_rx) = crossbeam_channel::bounded::<Vec<f32>>(4);

        let fwd_handle = std::thread::Builder::new()
            .name("audio-tee".into())
            .spawn(move || -> Vec<f32> {
                let mut all_samples: Vec<f32> = Vec::new();
                for chunk in audio_rx.iter() {
                    all_samples.extend_from_slice(&chunk);
                    if play_tx.send(chunk).is_err() {
                        break;
                    }
                }
                all_samples
            })?;

        audio::playback_loop(device, play_rx, source_rate, shutdown_p)?;

        let all_samples = fwd_handle
            .join()
            .map_err(|_| anyhow::anyhow!("Tee thread panicked"))?;

        save_wav(&save_path, &all_samples, source_rate)?;
        println!("Audio saved to {}", save_path.display());
    } else if want_play {
        let device = audio::get_output_device(args.device)?;
        let shutdown_p = shutdown.clone();
        audio::playback_loop(device, audio_rx, source_rate, shutdown_p)?;
    } else {
        let save_path = args.save.unwrap();
        let mut all_samples: Vec<f32> = Vec::new();
        for chunk in audio_rx.iter() {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            all_samples.extend_from_slice(&chunk);
        }
        save_wav(&save_path, &all_samples, source_rate)?;
        println!("Audio saved to {}", save_path.display());
    }

    synth_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Synthesis thread panicked"))?;

    println!("Done.");
    Ok(())
}

fn save_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s16)?;
    }
    writer.finalize()?;
    Ok(())
}
