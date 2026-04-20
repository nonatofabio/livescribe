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

/// Extract text from a PDF file, applying post-processing to fix common issues.
fn extract_pdf(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let raw_text = pdf_extract::extract_text_from_mem(&bytes)
        .context("Failed to extract text from PDF")?;

    let cleaned = clean_pdf_text(&raw_text);

    if cleaned.trim().is_empty() {
        eprintln!(
            "Warning: PDF appears to contain no extractable text (may be image-based)"
        );
    }

    Ok(cleaned)
}

/// Clean raw PDF extraction output:
/// - Strip "unknown glyph name" warnings pdf-extract leaks into stdout
/// - Normalize Unicode ligatures (ﬁ -> fi, ﬂ -> fl, etc.)
/// - Dehyphenate line-broken words ("pro-\ncess" -> "process")
/// - Collapse narrow-column single line breaks into spaces
pub fn clean_pdf_text(raw: &str) -> String {
    // Step 1: Strip pdf-extract warning lines that leak into stdout.
    let mut lines: Vec<&str> = raw
        .lines()
        .filter(|line| !line.starts_with("unknown glyph name "))
        .collect();

    // Trim trailing empty lines from the filter
    while lines.last().map_or(false, |l| l.is_empty()) {
        lines.pop();
    }

    let joined = lines.join("\n");

    // Step 2: Normalize Unicode ligatures to ASCII equivalents.
    // PDF extraction commonly produces these which confuse TTS engines.
    let normalized = joined
        .replace('\u{FB00}', "ff")   // ﬀ
        .replace('\u{FB01}', "fi")   // ﬁ
        .replace('\u{FB02}', "fl")   // ﬂ
        .replace('\u{FB03}', "ffi")  // ﬃ
        .replace('\u{FB04}', "ffl")  // ﬄ
        .replace('\u{FB05}', "st")   // ﬅ
        .replace('\u{FB06}', "st")   // ﬆ
        // Smart quotes -> plain
        .replace('\u{2018}', "'")    // '
        .replace('\u{2019}', "'")    // '
        .replace('\u{201C}', "\"")   // "
        .replace('\u{201D}', "\"")   // "
        // Dashes -> ASCII
        .replace('\u{2013}', "-")    // en dash
        .replace('\u{2014}', " - ")  // em dash (with spaces for natural pause)
        // Non-breaking space and other whitespace oddities
        .replace('\u{00A0}', " ")
        .replace('\u{2009}', " ")    // thin space
        .replace('\u{202F}', " ");   // narrow no-break

    // Step 3: Dehyphenate line breaks (e.g., "pro-\ncess" -> "process").
    // Only for lowercase letters to avoid merging legitimate hyphenated terms.
    let dehyphenated = dehyphenate(&normalized);

    // Step 4: Collapse single line breaks within paragraphs into spaces.
    // Preserve double line breaks (paragraph boundaries).
    collapse_line_breaks(&dehyphenated)
}

fn dehyphenate(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Look for pattern: [lowercase letter]-\n[lowercase letter]
        if i + 2 < chars.len()
            && chars[i].is_ascii_lowercase()
            && chars[i + 1] == '-'
            && chars[i + 2] == '\n'
            && i + 3 < chars.len()
            && chars[i + 3].is_ascii_lowercase()
        {
            out.push(chars[i]);
            // Skip the '-\n', next iteration pushes the lowercase letter
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }

    out
}

fn collapse_line_breaks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\n' {
            // Count consecutive newlines
            let mut count = 1;
            while let Some(&'\n') = chars.peek() {
                chars.next();
                count += 1;
            }
            if count >= 2 {
                // Paragraph boundary - preserve
                out.push_str("\n\n");
            } else {
                // Single line break within paragraph - replace with space
                // Avoid double spaces
                if !out.ends_with(' ') && !out.is_empty() {
                    out.push(' ');
                }
            }
        } else {
            out.push(ch);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_unknown_glyph_warnings() {
        let raw = "unknown glyph name 'C223' for font X\nunknown glyph name 'C223' for font X\n\nHello world";
        let cleaned = clean_pdf_text(raw);
        assert!(!cleaned.contains("unknown glyph"));
        assert!(cleaned.contains("Hello world"));
    }

    #[test]
    fn normalizes_ligatures() {
        let raw = "the signi\u{FB01}cance of re\u{FB02}ection";
        let cleaned = clean_pdf_text(raw);
        assert!(cleaned.contains("significance"));
        assert!(cleaned.contains("reflection"));
    }

    #[test]
    fn dehyphenates_line_breaks() {
        let raw = "the pro-\ncess of achieving homeo-\nstasis";
        let cleaned = clean_pdf_text(raw);
        assert!(cleaned.contains("process of"));
        assert!(cleaned.contains("homeostasis"));
    }

    #[test]
    fn preserves_paragraph_boundaries() {
        let raw = "First paragraph\nspans two lines.\n\nSecond paragraph\nhere.";
        let cleaned = clean_pdf_text(raw);
        assert!(cleaned.contains("First paragraph spans two lines."));
        assert!(cleaned.contains("\n\nSecond"));
    }

    #[test]
    fn preserves_legitimate_hyphens() {
        // Hyphenated terms not at line-end should stay intact
        let raw = "state-of-the-art methods are well-known";
        let cleaned = clean_pdf_text(raw);
        assert!(cleaned.contains("state-of-the-art"));
        assert!(cleaned.contains("well-known"));
    }
}
