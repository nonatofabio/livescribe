use anyhow::{bail, Context, Result};

const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-opus-4-0-20250514";

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

/// Rewrite document text for natural TTS narration using Claude via Bedrock.
///
/// Uses the Converse API with the specified model. Defaults to Claude Opus 4.
/// Requires AWS credentials in the environment (AWS_PROFILE, env vars, or IAM role).
pub fn rewrite_for_speech(text: &str, model_id: Option<&str>) -> Result<String> {
    let model = model_id.unwrap_or(DEFAULT_MODEL_ID);

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
