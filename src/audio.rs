use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};

use crate::AudioChunk;

/// Target sample rate for Whisper (always 16kHz mono).
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Lists all available audio input devices.
pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Available input devices:");

    let mut count = 0;
    for device in host.input_devices().context("Failed to enumerate input devices")? {
        let name = device.name().unwrap_or_else(|_| "Unknown".into());
        let config = device
            .default_input_config()
            .map(|c| {
                format!(
                    "{}Hz, {}ch",
                    c.sample_rate().0,
                    c.channels()
                )
            })
            .unwrap_or_else(|_| "unknown config".into());
        println!("  [{}] {} ({})", count, name, config);
        count += 1;
    }

    if count == 0 {
        println!("  (no input devices found)");
    }

    Ok(())
}

/// Returns the audio input device at the given index, or the default.
pub fn get_device(index: Option<usize>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    match index {
        Some(i) => host
            .input_devices()
            .context("Failed to enumerate input devices")?
            .nth(i)
            .ok_or_else(|| anyhow::anyhow!("No input device at index {}", i)),
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device found")),
    }
}

/// Returns the device name or "Unknown".
pub fn device_name(device: &cpal::Device) -> String {
    device.name().unwrap_or_else(|_| "Unknown".into())
}

/// Converts multi-channel audio to mono by averaging all channels.
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Resamples audio from `from_rate` to `to_rate` using linear interpolation.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;
        output.push(samples[idx0] * (1.0 - frac) + samples[idx1] * frac);
    }

    output
}

/// Continuously captures audio and sends overlapping chunks to the channel.
///
/// Uses the device's native sample rate and channel count, then converts
/// to 16kHz mono in the callback. Recording never blocks on transcription.
pub fn capture_loop(
    device: cpal::Device,
    chunk_duration_secs: u32,
    overlap_secs: u32,
    audio_tx: Sender<AudioChunk>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    // Use the device's preferred config instead of forcing 16kHz
    let supported = device
        .default_input_config()
        .context("Failed to get default input config")?;

    let device_rate = supported.sample_rate().0;
    let device_channels = supported.channels();
    let sample_format = supported.sample_format();

    eprintln!(
        "Audio device config: {}Hz, {}ch, {:?}",
        device_rate, device_channels, sample_format
    );

    let config: cpal::StreamConfig = supported.into();

    // Shared buffer receives 16kHz mono samples (already converted in callback)
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let buf = buffer.clone();
            let rate = device_rate;
            let ch = device_channels;
            device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono = to_mono(data, ch);
                    let resampled = resample(&mono, rate, WHISPER_SAMPLE_RATE);
                    let mut b = buf.lock().unwrap();
                    b.extend_from_slice(&resampled);
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let buf = buffer.clone();
            let rate = device_rate;
            let ch = device_channels;
            device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let mono = to_mono(&floats, ch);
                    let resampled = resample(&mono, rate, WHISPER_SAMPLE_RATE);
                    let mut b = buf.lock().unwrap();
                    b.extend_from_slice(&resampled);
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let buf = buffer.clone();
            let rate = device_rate;
            let ch = device_channels;
            device.build_input_stream(
                &config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> =
                        data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).collect();
                    let mono = to_mono(&floats, ch);
                    let resampled = resample(&mono, rate, WHISPER_SAMPLE_RATE);
                    let mut b = buf.lock().unwrap();
                    b.extend_from_slice(&resampled);
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?
        }
        fmt => bail!("Unsupported sample format: {:?}", fmt),
    };

    stream.play().context("Failed to start audio stream")?;

    // All chunk math is in 16kHz mono samples (post-conversion)
    let chunk_samples = chunk_duration_secs as usize * WHISPER_SAMPLE_RATE as usize;
    let overlap_samples = overlap_secs as usize * WHISPER_SAMPLE_RATE as usize;
    let stride_samples = chunk_samples - overlap_samples;

    let mut chunk_index: u64 = 0;
    let mut carryover: Vec<f32> = Vec::new();

    while !shutdown.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));

        let available = {
            let b = buffer.lock().unwrap();
            b.len()
        };

        let needed = if chunk_index == 0 {
            chunk_samples
        } else {
            stride_samples
        };

        if available >= needed {
            let new_samples: Vec<f32> = {
                let mut b = buffer.lock().unwrap();
                b.drain(..needed).collect()
            };

            let timestamp = chrono::Local::now();

            let (full_chunk, overlap_count) = if chunk_index == 0 {
                (new_samples, 0usize)
            } else {
                let mut combined = carryover.clone();
                combined.extend_from_slice(&new_samples);
                (combined, overlap_samples)
            };

            if full_chunk.len() >= overlap_samples {
                carryover = full_chunk[full_chunk.len() - overlap_samples..].to_vec();
            }

            let chunk = AudioChunk {
                samples: full_chunk,
                timestamp,
                chunk_index,
                overlap_samples: overlap_count,
            };

            match audio_tx.try_send(chunk) {
                Ok(_) => {
                    eprintln!(
                        "[Chunk {}] Recorded {}s of audio",
                        chunk_index, chunk_duration_secs
                    );
                }
                Err(crossbeam_channel::TrySendError::Full(_)) => {
                    eprintln!(
                        "[WARN] Transcription falling behind, dropping chunk {}",
                        chunk_index
                    );
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
            }

            chunk_index += 1;
        }
    }

    drop(stream);
    Ok(())
}

// ---------------------------------------------------------------------------
// Audio output (TTS playback)
// ---------------------------------------------------------------------------

/// Lists all available audio output devices.
pub fn list_output_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Available output devices:");

    let mut count = 0;
    for device in host
        .output_devices()
        .context("Failed to enumerate output devices")?
    {
        let name = device.name().unwrap_or_else(|_| "Unknown".into());
        let config = device
            .default_output_config()
            .map(|c| format!("{}Hz, {}ch", c.sample_rate().0, c.channels()))
            .unwrap_or_else(|_| "unknown config".into());
        println!("  [{}] {} ({})", count, name, config);
        count += 1;
    }

    if count == 0 {
        println!("  (no output devices found)");
    }

    Ok(())
}

/// Returns the audio output device at the given index, or the default.
pub fn get_output_device(index: Option<usize>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    match index {
        Some(i) => host
            .output_devices()
            .context("Failed to enumerate output devices")?
            .nth(i)
            .ok_or_else(|| anyhow::anyhow!("No output device at index {}", i)),
        None => host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No default output device found")),
    }
}

/// Plays audio received from a channel through the given output device.
///
/// Receives Vec<f32> audio chunks, resamples from `source_rate` to the device's
/// native rate, and feeds them through a cpal output stream. Blocks until all
/// audio is played or shutdown is signaled.
pub fn playback_loop(
    device: cpal::Device,
    audio_rx: Receiver<Vec<f32>>,
    source_rate: u32,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let supported = device
        .default_output_config()
        .context("Failed to get default output config")?;

    let device_rate = supported.sample_rate().0;
    let device_channels = supported.channels();

    eprintln!(
        "Output device config: {}Hz, {}ch",
        device_rate, device_channels
    );

    let config: cpal::StreamConfig = supported.into();

    // Ring buffer: we push resampled samples, cpal callback pulls them
    let buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));

    let buf_read = buffer.clone();
    let ch = device_channels;

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut buf = buf_read.lock().unwrap();
            for frame in data.chunks_exact_mut(ch as usize) {
                if let Some(sample) = buf.pop_front() {
                    // Duplicate mono sample to all output channels
                    for s in frame.iter_mut() {
                        *s = sample;
                    }
                } else {
                    // Silence when buffer is empty
                    for s in frame.iter_mut() {
                        *s = 0.0;
                    }
                }
            }
        },
        |err| eprintln!("Output stream error: {}", err),
        None,
    )?;

    stream.play().context("Failed to start output stream")?;

    // Feed loop: receive synthesized audio, resample, push to ring buffer
    for chunk in audio_rx.iter() {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        let resampled = resample(&chunk, source_rate, device_rate);

        let mut buf = buffer.lock().unwrap();
        buf.extend(resampled.iter());
        drop(buf);

        // Pace feeding: wait if buffer has more than 1 second of audio
        loop {
            let len = buffer.lock().unwrap().len();
            if len < device_rate as usize || shutdown.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    // Wait for buffer to drain before stopping
    loop {
        let remaining = buffer.lock().unwrap().len();
        if remaining == 0 || shutdown.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // Small tail to let the last samples reach the DAC
    std::thread::sleep(std::time::Duration::from_millis(300));

    drop(stream);
    Ok(())
}
