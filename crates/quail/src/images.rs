//! Image documents: a binary column of PNG images, such as the pages
//! `AI.PDF_PAGES` renders, as the tokens of a token store.
//!
//! An image of `w` by `h` pixels is `(w / 32) * (h / 32)` image tokens
//! between the model's vision start and end tokens. Each image token's
//! id is [`IMAGE_TOKEN_BASE`] or more, unique to the image's bytes, so
//! the planner counts, shares, and deduplicates image documents as it
//! does text, and two documents share a token only where they hold the
//! same image. The executor reads each image's pixels from the scan's
//! document column.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use arrow::array::{Array, ArrayRef, AsArray, Int32Builder, LargeListArray, LargeListBuilder};
use arrow::datatypes::DataType;
use quail_planner::constants::{IMAGE_TOKEN_BASE, IMAGE_TOKEN_PIXELS};
use quail_planner::error::CompileError;
use quail_planner::inputs::HfTokenizer;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

/// Whether a document column of this type holds images: a binary
/// column of PNGs, or a list column whose binary parts are each a PNG
/// or UTF-8 text.
pub(crate) fn holds_images(data_type: &DataType) -> bool {
    match data_type {
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => true,
        DataType::List(part) | DataType::LargeList(part) => matches!(
            part.data_type(),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        ),
        _ => false,
    }
}

/// The width and height a PNG's header gives, or None for other bytes.
pub(crate) fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || !bytes.starts_with(PNG_SIGNATURE) || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((width, height))
}

/// The first token id of each image a planner has seen, by its bytes.
#[derive(Debug, Default)]
pub(crate) struct ImageIds {
    by_content: HashMap<(u64, usize), i32>,
    used: i64,
}

impl ImageIds {
    /// The first id of an image of `count` tokens with these bytes: the
    /// same id for the same bytes.
    fn first_id(&mut self, bytes: &[u8], count: i64) -> Result<i32, CompileError> {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        let key = (hasher.finish(), bytes.len());
        if let Some(first) = self.by_content.get(&key) {
            return Ok(*first);
        }
        let first = i64::from(IMAGE_TOKEN_BASE) + self.used;
        if first + count > i64::from(i32::MAX) {
            return Err(CompileError::new(
                "the session has read more image tokens than it can number; \
                 open a new session",
            ));
        }
        self.used += count;
        let first = first as i32;
        self.by_content.insert(key, first);
        Ok(first)
    }
}

/// The token ids of each image in `values`, a binary column: the vision
/// start token, one id per image token, and the vision end token. A NULL
/// image has no tokens.
///
/// # Errors
///
/// The model has no vision tokens, or a value is not a PNG whose sides
/// are each a multiple of 32 pixels.
pub(crate) fn image_token_lists(
    values: &ArrayRef,
    tokenizer: &HfTokenizer,
    ids: &mut ImageIds,
) -> Result<LargeListArray, CompileError> {
    let special = |token: &str| {
        tokenizer.added_token_id(token).ok_or_else(|| {
            CompileError::new(format!(
                "the model's tokenizer has no {token}, so the model reads no images"
            ))
        })
    };
    let start = special("<|vision_start|>")? as i32;
    let end = special("<|vision_end|>")? as i32;
    let mut lists = LargeListBuilder::new(Int32Builder::new());
    // one part's tokens: a PNG's image tokens, or else its UTF-8 text's
    let mut part = |bytes: &[u8], tokens: &mut Int32Builder, text: bool| {
        if text && !bytes.starts_with(PNG_SIGNATURE) {
            let text = std::str::from_utf8(bytes).map_err(|_| {
                CompileError::new("a document part is neither a PNG nor UTF-8 text")
            })?;
            let found = tokenizer.encode(text).map_err(|error| {
                CompileError::new(format!("a document part does not tokenize: {error}"))
            })?;
            tokens.append_values(
                &found.iter().map(|id| *id as i32).collect::<Vec<_>>(),
                &vec![true; found.len()],
            );
            return Ok(());
        }
        let count = image_tokens(bytes)?;
        let first = ids.first_id(bytes, count)?;
        tokens.append_value(start);
        tokens.append_values(
            &(first..first + count as i32).collect::<Vec<_>>(),
            &vec![true; count as usize],
        );
        tokens.append_value(end);
        Ok::<(), CompileError>(())
    };
    let parts = |list: &ArrayRef| -> Result<Vec<Option<Vec<u8>>>, CompileError> {
        Ok(match list.data_type() {
            DataType::Binary => list
                .as_binary::<i32>()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            DataType::LargeBinary => list
                .as_binary::<i64>()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            DataType::BinaryView => list
                .as_binary_view()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            other => {
                return Err(CompileError::new(format!(
                    "an image column is binary, not {other}"
                )));
            }
        })
    };
    match values.data_type() {
        DataType::List(_) | DataType::LargeList(_) => {
            let lengths: Vec<Option<ArrayRef>> = match values.data_type() {
                DataType::List(_) => values.as_list::<i32>().iter().collect(),
                _ => values.as_list::<i64>().iter().collect(),
            };
            for document in lengths {
                if let Some(found) = document {
                    for bytes in parts(&found)?.into_iter().flatten() {
                        part(&bytes, lists.values(), true)?;
                    }
                }
                lists.append(true);
            }
        }
        _ => {
            for image in parts(values)? {
                if let Some(bytes) = image {
                    part(&bytes, lists.values(), false)?;
                }
                lists.append(true);
            }
        }
    }
    Ok(lists.finish())
}

/// The image tokens of a PNG: one per 32 by 32 pixels.
fn image_tokens(bytes: &[u8]) -> Result<i64, CompileError> {
    let Some((width, height)) = png_size(bytes) else {
        return Err(CompileError::new(
            "an image document must be a PNG, such as a page AI.PDF_PAGES renders",
        ));
    };
    if width == 0
        || height == 0
        || width % IMAGE_TOKEN_PIXELS != 0
        || height % IMAGE_TOKEN_PIXELS != 0
    {
        return Err(CompileError::new(format!(
            "an image document's sides must each be a multiple of {IMAGE_TOKEN_PIXELS} \
             pixels, as AI.PDF_PAGES renders them; one is {width} by {height}"
        )));
    }
    Ok(i64::from(width / IMAGE_TOKEN_PIXELS) * i64::from(height / IMAGE_TOKEN_PIXELS))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes
    }

    #[test]
    fn an_image_is_one_token_per_32_pixels_square() {
        assert_eq!(image_tokens(&png(64, 96)).unwrap(), 6);
        assert!(image_tokens(&png(60, 96)).is_err());
        assert!(image_tokens(b"not a png at all, longer than 24").is_err());
    }

    #[test]
    fn the_same_image_takes_the_same_ids_and_another_takes_new_ones() {
        let mut ids = ImageIds::default();
        let first = ids.first_id(&png(64, 64), 4).unwrap();
        assert_eq!(first, IMAGE_TOKEN_BASE);
        assert_eq!(ids.first_id(&png(64, 64), 4).unwrap(), first);
        assert_eq!(ids.first_id(&png(32, 64), 2).unwrap(), first + 4);
    }
}
