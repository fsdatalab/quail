//! The model's tokenizer: text to token ids and back, with each token's
//! bytes and character range.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Builder, LargeListArray, LargeListBuilder, StringArray};
use arrow::error::ArrowError;

/// A model's Hugging Face tokenizer, read from its `tokenizer.json`.
pub struct HfTokenizer {
    inner: tokenizers::pipeline::PipelineTokenizer,
    options: tokenizers::pipeline::EncodeOptions,
    /// The pre-tokenizer split Quail runs itself, for a pattern the
    /// tokenizers release does not, and the tokenizer of its pieces; None
    /// when the release runs the pattern.
    split: Option<(Splitter, tokenizers::pipeline::PipelineTokenizer)>,
    /// Each token id's bytes, None for an id the vocabulary does not hold.
    token_bytes: Vec<Option<Vec<u8>>>,
    /// The end-of-text token the model's `tokenizer_config.json` names.
    eos_token_id: Option<u32>,
}

/// Token ids of several texts, flat, with each token's character range
/// in its text and each text's first token.
#[derive(Debug, Default, PartialEq)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub offsets: Vec<(i64, i64)>,
    /// `boundaries[i]..boundaries[i + 1]` are text `i`'s tokens.
    pub boundaries: Vec<i64>,
}

fn external(error: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(error.to_string().into())
}

impl HfTokenizer {
    /// Read the tokenizer at `path`: a model directory holding
    /// `tokenizer.json`, or the file itself.
    ///
    /// # Errors
    ///
    /// The file does not read as a tokenizer.
    pub fn from_file(path: &Path) -> Result<Self, ArrowError> {
        let (file, directory) = if path.is_dir() {
            (path.join("tokenizer.json"), Some(path))
        } else {
            (path.to_path_buf(), path.parent())
        };
        let text = std::fs::read_to_string(&file)?;
        // a model's file is in the 0.x format, which 1.0 reads once converted
        let converted = tokenizers::canonicalize_str(&text).map_err(external)?;
        let options = tokenizers::pipeline::EncodeOptions::no_specials();
        let mut inner = tokenizers::from_json(&converted).map_err(external)?;
        let mut split = None;
        if inner.encode("a b", &options).wait().is_err()
            && let Some((without, mut splitter)) = Splitter::taken_from(&converted)?
        {
            let pieces = tokenizers::from_json(&without).map_err(external)?;
            // a text without combining marks splits the same under the
            // pattern without them, which the release runs natively
            match Splitter::markless(&converted)
                .and_then(|json| tokenizers::from_json(&json).ok())
                .filter(|found| found.encode("a b", &options).wait().is_ok())
            {
                Some(markless) => inner = markless,
                None => {
                    splitter.marks = None;
                    inner = tokenizers::from_json(&without).map_err(external)?;
                }
            }
            split = Some((splitter, pieces));
        }
        let mut tokenizer = Self {
            inner,
            options,
            split,
            token_bytes: Vec::new(),
            eos_token_id: None,
        };
        tokenizer.token_bytes = tokenizer.read_token_bytes();
        tokenizer.eos_token_id = directory
            .and_then(eos_token)
            .and_then(|token| tokenizer.added_token_id(&token));
        Ok(tokenizer)
    }

    /// The number of token ids.
    pub fn len(&self) -> usize {
        self.token_bytes.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.token_bytes.is_empty()
    }

    /// The end-of-text token the model's config names, or None.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// The token ids of one text, without special tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on the text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ArrowError> {
        Ok(self.encode_texts(&[text])?.pop().unwrap_or_default())
    }

    /// The token ids of each text, without special tokens.
    fn encode_texts(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, ArrowError> {
        let ids = |found: &tokenizers::pipeline::Encoding| {
            tokenizers::pipeline::PipelineToken::cast_slice(found.ids()).to_vec()
        };
        let native = |texts: Vec<&str>| -> Result<Vec<Vec<u32>>, ArrowError> {
            let encodings = (self.inner.encode(texts, &self.options).wait()).map_err(external)?;
            Ok(encodings.iter().map(ids).collect())
        };
        let Some((split, tokenizer)) = &self.split else {
            return native(texts.to_vec());
        };
        let (marked, plain): (Vec<usize>, Vec<usize>) =
            (0..texts.len()).partition(|&index| split.needs(texts[index]));
        let mut found = vec![Vec::new(); texts.len()];
        for (index, ids) in plain
            .iter()
            .zip(native(plain.iter().map(|&i| texts[i]).collect())?)
        {
            found[*index] = ids;
        }
        if marked.is_empty() {
            return Ok(found);
        }
        // BPE merges no tokens across pre-tokens, so each piece encodes alone
        let mut pieces: Vec<String> = Vec::new();
        let mut ends = Vec::with_capacity(marked.len());
        for &index in &marked {
            split.pieces(texts[index], &mut pieces).map_err(external)?;
            ends.push(pieces.len());
        }
        let inputs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let encodings = (tokenizer.encode(inputs, &self.options).wait()).map_err(external)?;
        let mut start = 0;
        for (&index, end) in marked.iter().zip(ends) {
            found[index] = encodings[start..end].iter().flat_map(ids).collect();
            start = end;
        }
        Ok(found)
    }

    /// The tokens of each text, as a large list array of int32 ids; a
    /// null text has no tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on a text.
    pub fn encode_batch(&self, texts: &StringArray) -> Result<LargeListArray, ArrowError> {
        let inputs: Vec<&str> = texts.iter().map(|text| text.unwrap_or("")).collect();
        let encodings = self.encode_texts(&inputs)?;
        let total = encodings.iter().map(Vec::len).sum();
        let mut builder =
            LargeListBuilder::with_capacity(Int32Builder::with_capacity(total), encodings.len());
        for encoding in &encodings {
            for &id in encoding {
                builder.values().append_value(id as i32);
            }
            builder.append(true);
        }
        Ok(builder.finish())
    }

    /// The tokens of each text with each token's character range: a token
    /// starts at the character holding its first byte and ends after the
    /// character holding its last.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on a text, or a text's tokens' bytes do not
    /// spell it, so no range could be read.
    pub fn encode_with_offsets(&self, texts: &[&str]) -> Result<Encoded, ArrowError> {
        let encodings = self.encode_texts(texts)?;
        let mut found = Encoded {
            boundaries: vec![0],
            ..Encoded::default()
        };
        for (text, ids) in texts.iter().zip(&encodings) {
            let ids = ids.as_slice();
            let ranges = self.character_ranges(text, ids).ok_or_else(|| {
                external("AI.EXTRACT requires token bytes that preserve the source text")
            })?;
            found.ids.extend_from_slice(ids);
            found.offsets.extend(ranges);
            found.boundaries.push(found.ids.len() as i64);
        }
        Ok(found)
    }

    /// Each text's tokens, and their character ranges as a large list of
    /// (start, end) pairs: null for a text whose tokens' bytes do not
    /// spell it, and for a null text, which has no tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails on a text.
    pub fn encode_batch_with_offsets(
        &self,
        texts: &StringArray,
    ) -> Result<(LargeListArray, LargeListArray), ArrowError> {
        use arrow::array::{FixedSizeListBuilder, Int64Builder};
        let tokens = self.encode_batch(texts)?;
        let mut ranges = LargeListBuilder::new(FixedSizeListBuilder::new(Int64Builder::new(), 2));
        for (index, text) in texts.iter().enumerate() {
            let ids: Vec<u32> = tokens
                .value(index)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .expect("int32 token ids")
                .values()
                .iter()
                .map(|id| *id as u32)
                .collect();
            match self.character_ranges(text.unwrap_or(""), &ids) {
                Some(found) => {
                    for (start, end) in found {
                        let pair = ranges.values();
                        pair.values().append_value(start);
                        pair.values().append_value(end);
                        pair.append(true);
                    }
                    ranges.append(true);
                }
                None => ranges.append(false),
            }
        }
        Ok((tokens, ranges.finish()))
    }

    /// The character range of each of a text's tokens: a token starts at
    /// the character holding its first byte and ends after the character
    /// holding its last. None when the tokens' bytes do not spell the
    /// text.
    fn character_ranges(&self, text: &str, ids: &[u32]) -> Option<Vec<(i64, i64)>> {
        let bytes = text.as_bytes();
        // the characters that start before each byte position
        let mut starts_before = Vec::with_capacity(bytes.len() + 1);
        let mut count = 0i64;
        starts_before.push(0);
        for &byte in bytes {
            count += i64::from(byte & 0xC0 != 0x80);
            starts_before.push(count);
        }
        let mut ranges = Vec::with_capacity(ids.len());
        let mut at = 0usize;
        for &id in ids {
            let piece = self
                .token_bytes
                .get(id as usize)
                .and_then(Option::as_deref)
                .unwrap_or_default();
            let end = at + piece.len();
            if bytes.get(at..end) != Some(piece) {
                return None;
            }
            let first = if piece.is_empty() {
                starts_before[at]
            } else {
                starts_before[at + 1] - 1
            };
            ranges.push((first, starts_before[end]));
            at = end;
        }
        (at == bytes.len()).then_some(ranges)
    }

    /// The id of an added token, such as `<|im_end|>`, or None.
    pub fn added_token_id(&self, token: &str) -> Option<u32> {
        self.inner.get_added_vocabulary().token_to_id(token)
    }

    /// Each token id's text in the tokenizer's own spelling, such as
    /// `Ġthe` in byte-level BPE, empty for an id the vocabulary does not
    /// hold.
    pub fn token_texts(&self) -> Vec<String> {
        (0..self.len() as u32)
            .map(|id| {
                self.inner
                    .decode_tokens(&[id], false)
                    .pop()
                    .unwrap_or_default()
            })
            .collect()
    }

    /// The bytes of each token id's text, None for an id the vocabulary
    /// does not hold.
    pub fn token_bytes(&self) -> &[Option<Vec<u8>>] {
        &self.token_bytes
    }

    /// The bytes of token ids' texts, joined.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|id| {
                self.token_bytes
                    .get(*id as usize)
                    .and_then(Option::as_deref)
            })
            .flatten()
            .copied()
            .collect()
    }

    /// The text of token ids.
    ///
    /// # Errors
    ///
    /// An id is outside the vocabulary.
    pub fn decode(&self, ids: &[u32]) -> Result<String, ArrowError> {
        self.inner.decode(ids, false).map_err(external)
    }

    /// The text of each token id list, without special tokens.
    ///
    /// # Errors
    ///
    /// An id is outside the vocabulary.
    pub fn decode_batch(&self, lists: &[Vec<u32>]) -> Result<Vec<String>, ArrowError> {
        lists
            .iter()
            .map(|ids| self.inner.decode(ids, true).map_err(external))
            .collect()
    }

    /// An added token is its text; a model token has the pre-tokenizer's
    /// spelling undone: byte-level BPE's characters for bytes, or
    /// SentencePiece's `▁` for a space and `<0xHH>` for a byte.
    fn read_token_bytes(&self) -> Vec<Option<Vec<u8>>> {
        // ids past this many consecutive misses are past the vocabulary
        const MISSES: u32 = 4096;
        let added = self.inner.get_added_vocabulary();
        let model = self.inner.get_model();
        let mut texts: Vec<Option<(bool, String)>> = Vec::new();
        let (mut id, mut misses) = (0u32, 0u32);
        while misses < MISSES {
            let found = match added.simple_id_to_token(id) {
                Some(text) => Some((true, text)),
                None => model.id_to_token(id).map(|text| (false, text)),
            };
            misses = if found.is_some() { 0 } else { misses + 1 };
            texts.push(found);
            id += 1;
        }
        let size = texts
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |last| last + 1);
        texts.truncate(size);
        let byte_level = texts
            .iter()
            .flatten()
            .any(|(is_added, text)| !is_added && text.starts_with('\u{120}'));
        let bytes = byte_level_bytes();
        texts
            .into_iter()
            .map(|found| {
                found.map(|(is_added, text)| {
                    if is_added {
                        text.into_bytes()
                    } else if byte_level {
                        text.chars()
                            .flat_map(|c| match bytes.get(&c) {
                                Some(&byte) => vec![byte],
                                None => c.to_string().into_bytes(),
                            })
                            .collect()
                    } else {
                        sentencepiece_bytes(&text)
                    }
                })
            })
            .collect()
    }
}

/// The end-of-text token's text the `tokenizer_config.json` in a model's
/// directory names.
fn eos_token(directory: &Path) -> Option<String> {
    let text = std::fs::read_to_string(directory.join("tokenizer_config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    match config.get("eos_token")? {
        serde_json::Value::String(token) => Some(token.clone()),
        other => other.get("content")?.as_str().map(str::to_string),
    }
}

/// Byte-level BPE's character for each byte, reversed: printable
/// Latin-1 bytes stand for themselves, and the rest map to code points
/// from 256 in byte order.
fn byte_level_bytes() -> std::collections::HashMap<char, u8> {
    let printable =
        |byte: u8| (b'!'..=b'~').contains(&byte) || (0xA1..=0xAC).contains(&byte) || byte >= 0xAE;
    let mut next = 256u32;
    (0..=255u8)
        .map(|byte| {
            let c = if printable(byte) {
                char::from(byte)
            } else {
                next += 1;
                char::from_u32(next - 1).expect("a code point below 512")
            };
            (c, byte)
        })
        .collect()
}

/// A SentencePiece token's bytes: `<0xHH>` is the byte, and `▁` a space.
fn sentencepiece_bytes(text: &str) -> Vec<u8> {
    if let Some(hex) = text
        .strip_prefix("<0x")
        .and_then(|rest| rest.strip_suffix('>'))
        && let Ok(byte) = u8::from_str_radix(hex, 16)
    {
        return vec![byte];
    }
    text.replace('\u{2581}', " ").into_bytes()
}

/// Each document's tokens as one array, for tests and callers that
/// build token lists themselves.
pub fn token_lists(lists: Vec<ArrayRef>) -> Result<LargeListArray, ArrowError> {
    let field = Arc::new(arrow::datatypes::Field::new_list_field(
        lists
            .first()
            .map_or(arrow::datatypes::DataType::Int32, |list| {
                list.data_type().clone()
            }),
        true,
    ));
    let lengths: Vec<usize> = lists.iter().map(|list| list.len()).collect();
    let offsets = arrow::buffer::OffsetBuffer::<i64>::from_lengths(lengths);
    let parts: Vec<&dyn Array> = lists.iter().map(|list| list.as_ref()).collect();
    let values = if parts.is_empty() {
        arrow::array::new_empty_array(field.data_type())
    } else {
        arrow::compute::concat(&parts)?
    };
    LargeListArray::try_new(field, offsets, values, None)
}

#[cfg(test)]
mod byte_tests {
    use super::{byte_level_bytes, sentencepiece_bytes};

    #[test]
    fn byte_level_characters_map_back_to_bytes() {
        let bytes = byte_level_bytes();
        assert_eq!(bytes.len(), 256);
        assert_eq!(bytes[&'\u{120}'], b' ');
        assert_eq!(bytes[&'\u{10a}'], b'\n');
        assert_eq!(bytes[&'a'], b'a');
    }

    #[test]
    fn sentencepiece_spaces_and_byte_tokens_map_to_bytes() {
        assert_eq!(sentencepiece_bytes("\u{2581}the"), b" the");
        assert_eq!(sentencepiece_bytes("<0x0A>"), b"\n");
    }
}

#[cfg(test)]
mod offset_tests {
    use super::HfTokenizer;

    fn fixture() -> HfTokenizer {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tokenizer");
        HfTokenizer::from_file(&path).expect("the fixture tokenizer")
    }

    #[test]
    fn each_token_covers_the_characters_its_bytes_spell() {
        let tokenizer = fixture();
        let texts = ["a café", "", "yes no"];
        let found = tokenizer.encode_with_offsets(&texts).expect("encoded");
        assert_eq!(found.boundaries.len(), texts.len() + 1);
        for (index, text) in texts.iter().enumerate() {
            let range = found.boundaries[index] as usize..found.boundaries[index + 1] as usize;
            let ids = &found.ids[range.clone()];
            assert_eq!(tokenizer.decode_bytes(ids), text.as_bytes());
            let chars: Vec<char> = text.chars().collect();
            let mut covered = String::new();
            let mut end = 0;
            for &(start, stop) in &found.offsets[range] {
                assert!(start >= 0 && start <= stop && stop as usize <= chars.len());
                if start >= end {
                    covered.extend(&chars[start as usize..stop as usize]);
                }
                end = end.max(stop);
            }
            assert_eq!(covered, *text);
        }
        // the two tokens of é's two bytes both cover é
        let cafe = &found.offsets[found.boundaries[0] as usize..found.boundaries[1] as usize];
        assert_eq!(cafe.last(), Some(&(5, 6)));
    }
}

/// A `tokenizer.json`'s regex pre-tokenizer split, run ahead of the
/// tokenizer: added tokens first, as Hugging Face matches them before it
/// normalizes, then NFC, then the pattern, keeping the text between
/// matches as pieces too (`Isolated`).
struct Splitter {
    pattern: fancy_regex::Regex,
    added: Option<regex::Regex>,
    nfc: bool,
    /// A combining mark, the one class the native pattern leaves out;
    /// None when every text takes the split.
    marks: Option<regex::Regex>,
}

impl Splitter {
    /// The tokenizer JSON without its regex `Split`, and that split; None
    /// when the JSON has no regex `Split` in a form Quail runs.
    fn taken_from(json: &str) -> Result<Option<(String, Self)>, ArrowError> {
        let mut value: serde_json::Value = serde_json::from_str(json).map_err(external)?;
        let is_split = |step: &serde_json::Value| {
            step["type"] == "Split"
                && step["pattern"]["Regex"].is_string()
                && step["invert"] != true
                && step["behavior"].as_str().unwrap_or("Isolated") == "Isolated"
        };
        let pattern = match value["pre_tokenizer"]["type"].as_str() {
            Some("Sequence") => {
                let Some(steps) = value["pre_tokenizer"]["pretokenizers"].as_array_mut() else {
                    return Ok(None);
                };
                let splits: Vec<usize> =
                    (0..steps.len()).filter(|&i| is_split(&steps[i])).collect();
                let [index] = splits.as_slice() else {
                    return Ok(None);
                };
                steps.remove(*index)["pattern"]["Regex"]
                    .as_str()
                    .map(str::to_string)
            }
            Some("Split") if is_split(&value["pre_tokenizer"]) => {
                let pattern = value["pre_tokenizer"]["pattern"]["Regex"]
                    .as_str()
                    .map(str::to_string);
                value["pre_tokenizer"] = serde_json::Value::Null;
                pattern
            }
            _ => None,
        };
        let Some(pattern) = pattern else {
            return Ok(None);
        };
        let mut contents: Vec<String> = value["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|token| token["content"].as_str().map(str::to_string))
            .collect();
        // the longest of two overlapping tokens matches, as Hugging Face's trie does
        contents.sort_by_key(|content| std::cmp::Reverse(content.len()));
        let added = (!contents.is_empty())
            .then(|| {
                let escaped: Vec<String> = contents.iter().map(|c| regex::escape(c)).collect();
                regex::Regex::new(&escaped.join("|"))
            })
            .transpose()
            .map_err(external)?;
        let nfc = value["normalizer"]["type"] == "NFC";
        let splitter = Self {
            pattern: fancy_regex::Regex::new(&pattern).map_err(external)?,
            added,
            nfc,
            marks: Some(regex::Regex::new(r"\p{M}").map_err(external)?),
        };
        Ok(Some((value.to_string(), splitter)))
    }

    /// The tokenizer JSON with combining marks taken out of its `Split`
    /// pattern, such as Qwen3.5's pattern made Qwen2's, or None.
    fn markless(json: &str) -> Option<String> {
        let mut value: serde_json::Value = serde_json::from_str(json).ok()?;
        let mut changed = false;
        let mut rewrite = |step: &mut serde_json::Value| {
            if let Some(pattern) = step["pattern"]["Regex"].as_str()
                && pattern.contains(r"\p{M}")
            {
                let markless = pattern
                    .replace(r"[\p{L}\p{M}]", r"\p{L}")
                    .replace(r"\p{M}", "");
                step["pattern"]["Regex"] = serde_json::Value::String(markless);
                changed = true;
            }
        };
        match value["pre_tokenizer"]["type"].as_str() {
            Some("Sequence") => value["pre_tokenizer"]["pretokenizers"]
                .as_array_mut()?
                .iter_mut()
                .for_each(&mut rewrite),
            Some("Split") => rewrite(&mut value["pre_tokenizer"]),
            _ => {}
        }
        changed.then(|| value.to_string())
    }

    /// Whether a text takes the split: it holds a combining mark, or the
    /// native pattern is unavailable.
    fn needs(&self, text: &str) -> bool {
        self.marks.as_ref().is_none_or(|marks| marks.is_match(text))
    }

    /// Append a text's pieces, in order, to `out`.
    fn pieces(&self, text: &str, out: &mut Vec<String>) -> Result<(), fancy_regex::Error> {
        let mut at = 0;
        if let Some(added) = &self.added {
            for found in added.find_iter(text) {
                self.split(&text[at..found.start()], out)?;
                out.push(found.as_str().to_string());
                at = found.end();
            }
        }
        self.split(&text[at..], out)
    }

    fn split(&self, text: &str, out: &mut Vec<String>) -> Result<(), fancy_regex::Error> {
        use unicode_normalization::UnicodeNormalization;
        if text.is_empty() {
            return Ok(());
        }
        let normal: String = if self.nfc {
            text.nfc().collect()
        } else {
            text.to_string()
        };
        let mut at = 0;
        for found in self.pattern.find_iter(&normal) {
            let found = found?;
            if found.start() > at {
                out.push(normal[at..found.start()].to_string());
            }
            if found.end() > found.start() {
                out.push(found.as_str().to_string());
            }
            at = found.end();
        }
        if at < normal.len() {
            out.push(normal[at..].to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    /// Qwen3.5's pre-tokenizer pattern, which the tokenizers release
    /// does not run.
    const QWEN3_5: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    #[test]
    fn a_split_keeps_added_tokens_whole_and_letters_with_their_marks() {
        let json = serde_json::json!({
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN3_5}, "behavior": "Isolated",
                 "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]},
            "added_tokens": [{"content": "<|im_end|>"}],
        });
        let (without, splitter) = Splitter::taken_from(&json.to_string()).unwrap().unwrap();
        let mut pieces = Vec::new();
        // a decomposed é composes before the split
        splitter
            .pieces("Cafe\u{301} it's<|im_end|>  12", &mut pieces)
            .unwrap();
        assert_eq!(
            pieces,
            ["Café", " it", "'s", "<|im_end|>", " ", " ", "1", "2"]
        );
        assert!(!without.contains("Split"));
        assert!(!splitter.needs("Café"));
        assert!(splitter.needs("Cafe\u{301}"));
    }

    #[test]
    fn qwen3_5s_pattern_without_combining_marks_is_qwen2s() {
        let json = serde_json::json!({"pre_tokenizer": {"type": "Split",
            "pattern": {"Regex": QWEN3_5}}});
        let markless: serde_json::Value =
            serde_json::from_str(&Splitter::markless(&json.to_string()).unwrap()).unwrap();
        assert_eq!(
            markless["pre_tokenizer"]["pattern"]["Regex"],
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        );
    }
}
