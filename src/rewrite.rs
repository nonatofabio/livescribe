use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};

const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-opus-4-6-v1";

/// Max characters per chunk sent to the LLM. Claude supports ~200k tokens
/// but very long inputs are slow. 15k chars (~4k tokens) keeps responses fast.
const CHUNK_MAX_CHARS: usize = 15_000;

const SYSTEM_PROMPT: &str = r##"You are an expert at adapting written documents for natural text-to-speech narration.

Your task: rewrite the given text so it sounds natural, engaging, and clear when read aloud by a TTS engine.

Rules:
1. ASCII art, diagrams, tables, and code blocks: Replace with a brief spoken description. For example, replace an ASCII diagram with "The diagram shows a pipeline flowing from audio capture to transcription to output." Never read raw ASCII art character by character.

2. Section headings: Convert to spoken transitions. For example, a heading like "Architecture" becomes "Now let's talk about the architecture."

3. Bullet points and lists: Convert to flowing prose or use "First... Second... Third..." phrasing.

4. URLs and file paths: Simplify. For example, a GitHub URL becomes "the GitHub repository." File paths like src/main.rs become "the main source file."

5. Technical formatting: Expand abbreviations on first use. Convert markdown emphasis to natural stress words like "importantly" or "notably."

6. Paragraph pauses: Insert [pause] on its own line between major sections or topic changes. This creates a natural breathing pause in the audio. Use [pause] sparingly, only between sections, not between every sentence.

7. Numbers and symbols: Write out as words where natural. "$10.5M" becomes "ten and a half million dollars." ">=3.11" becomes "three point eleven or higher."

8. Preserve content: Do not remove or summarize substantive content. The rewrite should cover everything in the original, just make it listenable.

9. Tone: Conversational but informative, like a knowledgeable person explaining the document to a colleague.

10. Output only the rewritten text. No preamble, no commentary, no markdown formatting.
"##;

const WRITING_TEXT: &str = "Rewriting for natural speech narration using AI ";

fn writing_animation(alive: Arc<AtomicBool>) {
    let pen = '\u{270D}';
    let max_width: usize = 48;
    let chars: Vec<char> = WRITING_TEXT.chars().collect();
    let mut pos: usize = 0;

    while alive.load(Ordering::SeqCst) {
        let display = if pos < max_width {
            let end = pos.min(chars.len());
            let text: String = chars[..end].iter().collect();
            format!("{}{}", text, pen)
        } else {
            let scroll = pos - max_width;
            let start = scroll % chars.len();
            let text: String = (0..max_width)
                .map(|i| chars[(start + i) % chars.len()])
                .collect();
            format!("{}{}", text, pen)
        };

        eprint!("\r  {}\x1b[K", display);
        let _ = std::io::stderr().flush();

        pos += 1;
        std::thread::sleep(std::time::Duration::from_millis(80));
    }

    eprint!("\r\x1b[K");
    let _ = std::io::stderr().flush();
}

/// Split text into chunks at paragraph boundaries, respecting max size.
fn chunk_text(text: &str) -> Vec<String> {
    if text.len() <= CHUNK_MAX_CHARS {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in text.split("\n\n") {
        if !current.is_empty() && current.len() + paragraph.len() + 2 > CHUNK_MAX_CHARS {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

/// Rewrite document text for natural TTS narration using Claude via Bedrock.
///
/// Large documents are split into chunks and processed sequentially.
/// Shows a writing animation unless verbose mode prints debug logs instead.
pub fn rewrite_for_speech(text: &str, model_id: Option<&str>, verbose: bool) -> Result<String> {
    let model = model_id.unwrap_or(DEFAULT_MODEL_ID);
    let chunks = chunk_text(text);
    let total_chunks = chunks.len();

    if verbose {
        eprintln!(
            "[rewrite] Input: {} chars, split into {} chunk(s), model: {}",
            text.len(),
            total_chunks,
            model
        );
    }

    // Only show animation in non-verbose mode
    let alive = Arc::new(AtomicBool::new(!verbose));
    let anim_handle = if !verbose {
        let alive_clone = alive.clone();
        Some(
            std::thread::Builder::new()
                .name("rewrite-anim".into())
                .spawn(move || writing_animation(alive_clone))?,
        )
    } else {
        None
    };

    let mut all_results = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if verbose {
            eprintln!(
                "[rewrite] Chunk {}/{}: {} chars, sending to Bedrock...",
                i + 1,
                total_chunks,
                chunk.len()
            );
        }

        let start = Instant::now();
        let result = do_rewrite(chunk, model, verbose)?;
        let elapsed = start.elapsed();

        if verbose {
            eprintln!(
                "[rewrite] Chunk {}/{}: got {} chars back in {:.1}s",
                i + 1,
                total_chunks,
                result.len(),
                elapsed.as_secs_f64()
            );
        }

        all_results.push(result);
    }

    // Stop animation
    alive.store(false, Ordering::SeqCst);
    if let Some(handle) = anim_handle {
        let _ = handle.join();
    }

    Ok(all_results.join("\n\n[pause]\n\n"))
}

fn do_rewrite(text: &str, model: &str, verbose: bool) -> Result<String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create async runtime")?;

    rt.block_on(async {
        if verbose {
            eprintln!("[rewrite] Loading AWS config...");
        }

        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;

        if verbose {
            eprintln!("[rewrite] AWS region: {:?}", config.region());
        }

        let client = aws_sdk_bedrockruntime::Client::new(&config);

        if verbose {
            eprintln!("[rewrite] Calling converse API...");
        }

        let response = client
            .converse()
            .model_id(model)
            .system(
                aws_sdk_bedrockruntime::types::SystemContentBlock::Text(
                    SYSTEM_PROMPT.to_string(),
                ),
            )
            .messages(
                aws_sdk_bedrockruntime::types::Message::builder()
                    .role(aws_sdk_bedrockruntime::types::ConversationRole::User)
                    .content(
                        aws_sdk_bedrockruntime::types::ContentBlock::Text(text.to_string()),
                    )
                    .build()
                    .map_err(|e| anyhow::anyhow!("Failed to build message: {}", e))?,
            )
            .send()
            .await
            .context("Bedrock API call failed. Check your AWS credentials and region.")?;

        if verbose {
            if let Some(usage) = response.usage() {
                eprintln!(
                    "[rewrite] Token usage: input={}, output={}",
                    usage.input_tokens(),
                    usage.output_tokens()
                );
            }
            eprintln!(
                "[rewrite] Stop reason: {:?}",
                response.stop_reason()
            );
        }

        let output = response
            .output()
            .ok_or_else(|| anyhow::anyhow!("No output from Bedrock"))?;

        match output {
            aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) => {
                let mut result = String::new();
                for block in msg.content() {
                    if let aws_sdk_bedrockruntime::types::ContentBlock::Text(t) = block {
                        result.push_str(t);
                    }
                }
                if result.is_empty() {
                    bail!("Empty response from Bedrock");
                }
                Ok(result)
            }
            _ => bail!("Unexpected response type from Bedrock"),
        }
    })
}
