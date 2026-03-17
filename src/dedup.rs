use crate::transcribe::Segment;

/// Filters out segments that fall entirely within the overlap region.
///
/// Segments whose start time is before the overlap boundary are dropped
/// unless they extend significantly past it.
pub fn filter_overlap_segments<'a>(
    segments: &'a [Segment],
    overlap_samples: usize,
    sample_rate: u32,
) -> Vec<&'a Segment> {
    if overlap_samples == 0 {
        return segments.iter().collect();
    }

    let overlap_cs = (overlap_samples as i64 * 100) / sample_rate as i64;

    segments
        .iter()
        .filter(|s| {
            // Keep segments that start at or after the overlap boundary,
            // or that extend well past it (straddling segments).
            s.start_cs >= overlap_cs || s.end_cs > overlap_cs + 50
        })
        .collect()
}

/// Removes duplicate text at the boundary between consecutive transcriptions.
///
/// Uses word-level suffix-prefix matching: finds the longest run of matching
/// words between the end of `prev_text` and the start of `curr_text`, then
/// strips the matched prefix from `curr_text`.
pub fn deduplicate(prev_text: &str, curr_text: &str, overlap_ratio: f64) -> String {
    if prev_text.is_empty() || curr_text.is_empty() {
        return curr_text.to_string();
    }

    let prev_words: Vec<&str> = prev_text.split_whitespace().collect();
    let curr_words: Vec<&str> = curr_text.split_whitespace().collect();

    if prev_words.is_empty() || curr_words.is_empty() {
        return curr_text.to_string();
    }

    // Search window: look at the tail of prev_text proportional to overlap ratio
    let search_window = ((prev_words.len() as f64) * overlap_ratio * 1.5)
        .ceil()
        .max(3.0)
        .min(prev_words.len() as f64) as usize;

    let prev_suffix = &prev_words[prev_words.len() - search_window..];

    let mut best_match_len = 0;

    // Try every possible suffix of prev_suffix as a prefix of curr_words
    for start in 0..prev_suffix.len() {
        let candidate = &prev_suffix[start..];
        let compare_len = candidate.len().min(curr_words.len());

        if compare_len == 0 {
            continue;
        }

        let mut matching = 0;
        for j in 0..compare_len {
            if words_match(candidate[j], curr_words[j]) {
                matching += 1;
            } else {
                break;
            }
        }

        // Require at least 2 consecutive matches to avoid false positives
        if matching >= 2 && matching > best_match_len {
            best_match_len = matching;
        }
    }

    if best_match_len > 0 {
        curr_words[best_match_len..].join(" ")
    } else {
        curr_text.to_string()
    }
}

/// Fuzzy word comparison: case-insensitive, strips punctuation.
///
/// Whisper often produces slightly different punctuation for the same audio,
/// so we normalize before comparing.
fn words_match(a: &str, b: &str) -> bool {
    let clean_a: String = a
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    let clean_b: String = b
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    !clean_a.is_empty() && clean_a == clean_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_overlap() {
        let result = deduplicate("hello world", "foo bar", 0.25);
        assert_eq!(result, "foo bar");
    }

    #[test]
    fn test_exact_overlap() {
        let result = deduplicate(
            "the quick brown fox jumps over",
            "fox jumps over the lazy dog",
            0.5,
        );
        assert_eq!(result, "the lazy dog");
    }

    #[test]
    fn test_empty_prev() {
        let result = deduplicate("", "hello world", 0.25);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_empty_curr() {
        let result = deduplicate("hello world", "", 0.25);
        assert_eq!(result, "");
    }

    #[test]
    fn test_punctuation_tolerance() {
        let result = deduplicate(
            "and so, the story goes",
            "the story goes. And it was good.",
            0.5,
        );
        assert_eq!(result, "And it was good.");
    }

    #[test]
    fn test_words_match_basic() {
        assert!(words_match("Hello", "hello"));
        assert!(words_match("world!", "world"));
        assert!(words_match("it's", "its"));
        assert!(!words_match("hello", "world"));
    }
}
