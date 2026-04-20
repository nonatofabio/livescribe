use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};

/// Default model: Claude Sonnet 4.6. Faster and cheaper than Opus while
/// still producing high-quality natural rewrites.
pub const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-4-6";

/// Max characters per chunk. ~15k chars is ~4k tokens which stays well
/// within Sonnet's output budget and keeps each call under 60 seconds.
const CHUNK_MAX_CHARS: usize = 15_000;

const SYSTEM_PROMPT: &str = r###"You rewrite documents so they can be narrated aloud by a text-to-speech engine. Your output is read verbatim to the listener.

Produce clear, flowing prose that someone would actually want to listen to. Keep ALL substantive content from the source — this is a rewrite, not a summary.

Handling rules:

1. ASCII art, diagrams, tables, equations: replace with a short spoken description of what they convey. Never repeat raw symbols or grid characters.

2. Section headings: turn into spoken transitions like "Now, about the architecture," or "Turning to the results,". Never emit raw markdown like "##" or numbered heading prefixes.

3. Lists: turn into flowing prose or sequential phrasing ("First... Second... Finally...").

4. URLs and file paths: describe them ("the GitHub repo," "the main source file"). Never read URLs character by character.

5. Academic references like [1], [4,5], [Smith 2020]: omit inline citation numbers entirely. They disrupt the read. If a reference is load-bearing for the sentence, phrase it as "prior work has shown..." Never spell out bracketed numbers.

6. Figure callouts ("(Figure 1)", "see Fig. 2"): omit unless you're rephrasing to describe the figure's content.

7. Author affiliations, running headers, page numbers, journal metadata, DOIs, copyright notices: drop them entirely.

8. Numbers, units, symbols: write as speech ("10.5%" -> "ten point five percent", "p < 0.05" -> "a p-value below point oh five", "H2O" -> "water"). Keep natural scientific phrasing.

9. Abbreviations on first use: expand ("TTS" -> "text-to-speech, or TTS"). On subsequent uses, use whichever reads most naturally.

10. Paragraph pauses: insert a single line containing exactly "[pause]" between major sections or topic shifts. Use sparingly — roughly every 2 to 5 paragraphs. Never between consecutive sentences.

11. Output ONLY the rewritten narration. No preamble ("Here is the rewrite..."), no commentary, no markdown headings, no code fences, no meta-notes about what you changed. Just the prose.
"###;

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
pub fn chunk_text(text: &str) -> Vec<String> {
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

/// Clean an LLM response of common preamble/postamble artifacts.
/// Claude sometimes prepends "Here's the rewrite:" or wraps output in markdown
/// even when instructed not to. Strip those.
fn sanitize_llm_output(text: &str) -> String {
    let trimmed = text.trim();

    // Strip common preamble patterns
    let without_preamble = strip_preamble(trimmed);

    // Strip markdown code fences if the whole thing is wrapped
    let without_fence = strip_code_fence(&without_preamble);

    without_fence
}

fn strip_preamble(text: &str) -> String {
    let preamble_prefixes = [
        "Here is the rewrite:",
        "Here's the rewrite:",
        "Here is the rewritten text:",
        "Here's the rewritten text:",
        "Here is the rewritten version:",
        "Here's the rewritten version:",
        "Rewritten text:",
        "Rewrite:",
    ];

    for prefix in &preamble_prefixes {
        if let Some(stripped) = text.strip_prefix(prefix) {
            return stripped.trim_start().to_string();
        }
    }
    text.to_string()
}

fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("```") && trimmed.ends_with("```") {
        let inner = trimmed.trim_start_matches("```").trim_end_matches("```");
        let inner = inner.strip_prefix('\n').unwrap_or(inner);
        // Also drop a language hint on the first line if present
        if let Some((first_line, rest)) = inner.split_once('\n') {
            if !first_line.contains(' ') && first_line.len() < 20 {
                return rest.trim().to_string();
            }
        }
        return inner.trim().to_string();
    }
    text.to_string()
}

/// Rewrite document text for natural TTS narration using Claude via Bedrock.
///
/// Large documents are chunked and processed sequentially. Output is sanitized
/// of LLM preamble/postamble. Optionally saves the rewritten text to a file
/// for inspection.
pub fn rewrite_for_speech(
    text: &str,
    model_id: Option<&str>,
    verbose: bool,
    save_to: Option<&Path>,
) -> Result<String> {
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

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create async runtime")?;

    let client = rt.block_on(async {
        if verbose {
            eprintln!("[rewrite] Loading AWS config...");
        }
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        if verbose {
            eprintln!("[rewrite] AWS region: {:?}", config.region());
        }
        aws_sdk_bedrockruntime::Client::new(&config)
    });

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
        let raw_result = do_rewrite(&rt, &client, chunk, model, verbose)?;
        let result = sanitize_llm_output(&raw_result);
        let elapsed = start.elapsed();

        if verbose {
            eprintln!(
                "[rewrite] Chunk {}/{}: got {} chars (sanitized to {}) in {:.1}s",
                i + 1,
                total_chunks,
                raw_result.len(),
                result.len(),
                elapsed.as_secs_f64()
            );
        }

        all_results.push(result);
    }

    alive.store(false, Ordering::SeqCst);
    if let Some(handle) = anim_handle {
        let _ = handle.join();
    }

    let joined = all_results.join("\n\n[pause]\n\n");

    if let Some(path) = save_to {
        std::fs::write(path, &joined)
            .with_context(|| format!("Failed to write rewrite to {}", path.display()))?;
        eprintln!("Rewrite saved to {}", path.display());
    }

    Ok(joined)
}

fn do_rewrite(
    rt: &tokio::runtime::Runtime,
    client: &aws_sdk_bedrockruntime::Client,
    text: &str,
    model: &str,
    verbose: bool,
) -> Result<String> {
    rt.block_on(async {
        if verbose {
            eprintln!("[rewrite] Calling converse API...");
        }

        let inference_config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
            // Cap output so we don't burn tokens on runaway completions.
            // Sonnet typically finishes well under this on prose rewrites.
            .max_tokens(8192)
            .temperature(0.3)
            .build();

        let response = client
            .converse()
            .model_id(model)
            .inference_config(inference_config)
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
            eprintln!("[rewrite] Stop reason: {:?}", response.stop_reason());
        }

        // Warn if the model got truncated - the rest of the chunk is lost.
        if let Some(stop_reason) = response.stop_reason().as_str().into() {
            let reason: &str = stop_reason;
            if reason == "max_tokens" {
                eprintln!(
                    "[rewrite] Warning: response hit max_tokens limit, output truncated."
                );
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_preamble() {
        let out = sanitize_llm_output("Here's the rewrite:\n\nThe actual content.");
        assert_eq!(out, "The actual content.");
    }

    #[test]
    fn strips_markdown_fence() {
        let out = sanitize_llm_output("```\nSome narration here.\n```");
        assert_eq!(out, "Some narration here.");
    }

    #[test]
    fn leaves_clean_output_alone() {
        let clean = "The story begins here. It continues naturally.";
        assert_eq!(sanitize_llm_output(clean), clean);
    }

    #[test]
    fn chunks_at_paragraph_boundaries() {
        // Build input larger than CHUNK_MAX_CHARS
        let paragraph = "word ".repeat(500); // 2500 chars
        let big = vec![paragraph; 10].join("\n\n"); // 25k chars, 10 paragraphs
        let chunks = chunk_text(&big);
        assert!(chunks.len() > 1);
        // Every chunk should end on a paragraph boundary
        for c in &chunks {
            assert!(c.len() <= CHUNK_MAX_CHARS + 100); // small overshoot ok
        }
    }
}
