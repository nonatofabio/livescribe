use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};

const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-opus-4-6-v1";

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

/// The writing animation text that scrolls under the pen.
const WRITING_TEXT: &str = "Rewriting for natural speech narration using AI ";

/// Runs a writing animation on stderr while `alive` is true.
/// Shows text growing character by character with a pen at the end,
/// then scrolling left once it hits max width.
fn writing_animation(alive: Arc<AtomicBool>) {
    let pen = '\u{270D}'; // ✍
    let max_width: usize = 48;
    let chars: Vec<char> = WRITING_TEXT.chars().collect();
    let mut pos: usize = 0;

    while alive.load(Ordering::SeqCst) {
        let display = if pos < max_width {
            // Growing phase: text appears character by character
            let end = pos.min(chars.len());
            let text: String = chars[..end].iter().collect();
            format!("{}{}", text, pen)
        } else {
            // Scrolling phase: text slides left, pen stays at right edge
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

    // Clear the animation line
    eprint!("\r\x1b[K");
    let _ = std::io::stderr().flush();
}

/// Rewrite document text for natural TTS narration using Claude via Bedrock.
///
/// Shows a writing animation while the API call is in-flight.
/// Uses the Converse API with the specified model. Defaults to Claude Opus 4.6.
/// Requires AWS credentials in the environment (AWS_PROFILE, env vars, or IAM role).
pub fn rewrite_for_speech(text: &str, model_id: Option<&str>) -> Result<String> {
    let model = model_id.unwrap_or(DEFAULT_MODEL_ID);

    // Start the writing animation
    let alive = Arc::new(AtomicBool::new(true));
    let alive_clone = alive.clone();
    let anim_handle = std::thread::Builder::new()
        .name("rewrite-anim".into())
        .spawn(move || writing_animation(alive_clone))?;

    let result = do_rewrite(text, model);

    // Stop animation
    alive.store(false, Ordering::SeqCst);
    let _ = anim_handle.join();

    result
}

fn do_rewrite(text: &str, model: &str) -> Result<String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create async runtime")?;

    rt.block_on(async {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = aws_sdk_bedrockruntime::Client::new(&config);

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
