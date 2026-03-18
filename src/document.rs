use std::path::Path;

use anyhow::{bail, Context, Result};

/// Extract plain text from a supported document file.
pub fn extract_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "txt" => std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display())),
        "md" | "markdown" => extract_markdown(path),
        "pdf" => extract_pdf(path),
        _ => bail!(
            "Unsupported file format '.{}'. Supported: .txt, .md, .pdf",
            ext
        ),
    }
}

/// Strip Markdown formatting and extract readable plain text.
fn extract_markdown(path: &Path) -> Result<String> {
    let md = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    use pulldown_cmark::{Event, Parser, Tag, TagEnd};

    let parser = Parser::new(&md);
    let mut text = String::new();
    let mut in_code_block = false;

    for event in parser {
        match event {
            Event::Text(t) if !in_code_block => {
                text.push_str(&t);
            }
            Event::SoftBreak | Event::HardBreak => {
                text.push(' ');
            }
            Event::Start(Tag::CodeBlock(_)) => {
                in_code_block = true;
            }
            Event::End(TagEnd::CodeBlock) => {
                in_code_block = false;
            }
            Event::Start(Tag::Heading { .. }) => {
                text.push('\n');
            }
            Event::End(TagEnd::Heading(_)) => {
                text.push_str(". ");
            }
            Event::End(TagEnd::Paragraph) => {
                text.push('\n');
            }
            Event::Start(Tag::Item) => {
                text.push_str(". ");
            }
            _ => {}
        }
    }

    Ok(text)
}

/// Extract text from a PDF file.
fn extract_pdf(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let text = pdf_extract::extract_text_from_mem(&bytes)
        .context("Failed to extract text from PDF")?;

    if text.trim().is_empty() {
        eprintln!(
            "Warning: PDF appears to contain no extractable text (may be image-based)"
        );
    }

    Ok(text)
}
