//! Image documents: a binary column of PNG or JPEG images, such as the
//! pages `AI.PDF_PAGES` renders, as the tokens of a token store.
//!
//! The model reads an image at the size [`image_size`] gives, as Qwen's
//! image processor resizes it. An image of `w` by `h` pixels at that
//! size is `(w / t) * (h / t)` image tokens, where `t` is the model's
//! [`ImageSizing::token_pixels`],
//! between the model's vision start and end tokens. Each image token's
//! id is [`IMAGE_TOKEN_BASE`] or more, unique to the image's bytes, so
//! the planner counts, shares, and deduplicates image documents as it
//! does text, and two documents share a token only where they hold the
//! same image. The executor reads each image's pixels from the scan's
//! document column.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int32Builder, LargeListArray, LargeListBuilder};
use arrow::datatypes::DataType;
use quail_planner::constants::IMAGE_TOKEN_BASE;
use quail_planner::error::CompileError;
use quail_planner::inputs::HfTokenizer;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const JPEG_SIGNATURE: &[u8] = b"\xff\xd8\xff";
/// The longest an image's longer side may be, as a multiple of its
/// shorter side, as Qwen's `smart_resize` allows.
const MAX_ASPECT: u32 = 200;

/// How a model's image processor sizes an image, from the model's spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSizing {
    /// The pixels along each side of one image token: the patch's side
    /// times the patches merged along a side.
    pub token_pixels: u32,
    /// The fewest pixels the model reads an image at.
    pub min_pixels: u64,
    /// The most pixels the model reads an image at.
    pub max_pixels: u64,
}

impl ImageSizing {
    /// The sizing a model spec, as JSON, gives; None when the model
    /// loads no vision encoder.
    pub(crate) fn of_model(model: &serde_json::Value) -> Option<Self> {
        let field = |name: &str| model[name].as_u64().unwrap_or(0);
        // a spec that loads only some of its checkpoint's layers, or a
        // decision model, loads no vision encoder
        if !model["vision_tower"].as_bool().unwrap_or(false)
            || model["role"].as_str() == Some("decision")
            || field("checkpoint_layers") != 0
        {
            return None;
        }
        let token_pixels =
            u32::try_from(field("image_patch_pixels") * field("image_merge")).ok()?;
        let (min_pixels, max_pixels) = (field("image_min_pixels"), field("image_max_pixels"));
        (token_pixels > 0 && max_pixels > 0).then_some(Self {
            token_pixels,
            min_pixels,
            max_pixels,
        })
    }
}
/// The field of a page struct that holds its PNG, as AI.PDF_PAGES names it.
const PAGE_IMAGE: &str = "image";

/// Whether a document column of this type holds images: a binary
/// column of PNGs, or a list column whose binary parts are each a PNG
/// or UTF-8 text.
pub(crate) fn holds_images(data_type: &DataType) -> bool {
    match data_type {
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => true,
        DataType::List(part) | DataType::LargeList(part) => match part.data_type() {
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => true,
            // the pages AI.PDF_PAGES returns: each its number and its image
            DataType::Struct(fields) => fields
                .find(PAGE_IMAGE)
                .is_some_and(|(_, field)| holds_images(field.data_type())),
            _ => false,
        },
        _ => false,
    }
}

/// The width and height a PNG's header gives, or None for other bytes.
fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || !bytes.starts_with(PNG_SIGNATURE) || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((width, height))
}

/// The width and height a JPEG's frame header gives, or None for other
/// bytes.
fn jpeg_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if !bytes.starts_with(JPEG_SIGNATURE) {
        return None;
    }
    let mut at = 2;
    while at + 4 <= bytes.len() {
        if bytes[at] != 0xff {
            return None;
        }
        let marker = bytes[at + 1];
        // fill bytes before a marker
        if marker == 0xff {
            at += 1;
            continue;
        }
        // markers with no length: restarts, and the start of the image
        if (0xd0..=0xd8).contains(&marker) || marker == 0x01 {
            at += 2;
            continue;
        }
        let length = usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]));
        // a start-of-frame marker; 0xc4, 0xc8, and 0xcc are other tables
        if (0xc0..=0xcf).contains(&marker) && ![0xc4, 0xc8, 0xcc].contains(&marker) {
            let frame = bytes.get(at + 5..at + 9)?;
            let height = u32::from(u16::from_be_bytes([frame[0], frame[1]]));
            let width = u32::from(u16::from_be_bytes([frame[2], frame[3]]));
            return Some((width, height));
        }
        at += 2 + length;
    }
    None
}

/// The width and height the model reads an image at: each side a
/// multiple of the sizing's token pixels, and the area from its minimum
/// to its maximum, as Qwen's `smart_resize` gives them.
///
/// # Errors
///
/// The bytes are not a PNG or a JPEG, or the image has no pixels or is
/// over 200 times as long as it is wide.
pub fn image_size(bytes: &[u8], sizing: ImageSizing) -> Result<(u32, u32), CompileError> {
    let Some((width, height)) = png_size(bytes).or_else(|| jpeg_size(bytes)) else {
        return Err(CompileError::new(
            "an image document must be a PNG or a JPEG, such as a page AI.PDF_PAGES renders",
        ));
    };
    if width == 0 || height == 0 || width.max(height) > MAX_ASPECT * width.min(height) {
        return Err(CompileError::new(format!(
            "an image document must have a side at most {MAX_ASPECT} times its other \
             side; one is {width} by {height}"
        )));
    }
    let factor = f64::from(sizing.token_pixels);
    let (min_pixels, max_pixels) = (sizing.min_pixels as f64, sizing.max_pixels as f64);
    let (width, height) = (f64::from(width), f64::from(height));
    // Python's round(), which rounds a half to the even multiple
    let mut sides =
        [height, width].map(|side| factor.max((side / factor).round_ties_even() * factor));
    if sides[0] * sides[1] > max_pixels {
        let beta = (height * width / max_pixels).sqrt();
        sides = [height, width].map(|side| factor.max((side / beta / factor).floor() * factor));
    } else if sides[0] * sides[1] < min_pixels {
        let beta = (min_pixels / (height * width)).sqrt();
        sides = [height, width].map(|side| (side * beta / factor).ceil() * factor);
    }
    Ok((sides[1] as u32, sides[0] as u32))
}

/// Whether a document part's bytes are an image rather than text.
fn is_image(bytes: &[u8]) -> bool {
    bytes.starts_with(PNG_SIGNATURE) || bytes.starts_with(JPEG_SIGNATURE)
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
/// The model has no vision tokens, or a value is not an image
/// [`image_size`] reads.
pub(crate) fn image_token_lists(
    values: &ArrayRef,
    tokenizer: &HfTokenizer,
    ids: &mut ImageIds,
    sizing: ImageSizing,
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
    // one part's tokens: an image's tokens, or else its UTF-8 text's
    let mut part = |bytes: &[u8], tokens: &mut Int32Builder, text: bool| {
        if text && !is_image(bytes) {
            let text = std::str::from_utf8(bytes).map_err(|_| {
                CompileError::new("a document part is neither an image nor UTF-8 text")
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
        let count = image_tokens(bytes, sizing)?;
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
        let list = match list.data_type() {
            DataType::Struct(_) => list
                .as_struct()
                .column_by_name(PAGE_IMAGE)
                .cloned()
                .ok_or_else(|| CompileError::new("a page has no image field"))?,
            _ => Arc::clone(list),
        };
        let list = &list;
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

/// The image tokens of an image: one per token's square of pixels, at
/// the size the model reads it.
fn image_tokens(bytes: &[u8], sizing: ImageSizing) -> Result<i64, CompileError> {
    let (width, height) = image_size(bytes, sizing)?;
    let side = sizing.token_pixels;
    Ok(i64::from(width / side) * i64::from(height / side))
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

    /// A JPEG's start, an APP0 segment, and a baseline frame header.
    fn jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46];
        bytes.extend_from_slice(&[0xff, 0xc0, 0x00, 0x11, 0x08]);
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes
    }

    /// Qwen3.5's image processor: 16-pixel patches, merged 2 by 2.
    const QWEN: ImageSizing = ImageSizing {
        token_pixels: 32,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
    };

    #[test]
    fn an_image_is_one_token_per_32_pixels_square() {
        assert_eq!(image_tokens(&png(320, 256), QWEN).unwrap(), 80);
        assert_eq!(image_tokens(&jpeg(320, 256), QWEN).unwrap(), 80);
        assert!(image_tokens(b"not an image at all, longer than 24", QWEN).is_err());
        assert!(image_tokens(&png(64, 64 * 201), QWEN).is_err());
    }

    #[test]
    fn a_model_with_other_patches_sizes_an_image_its_own_way() {
        // 14-pixel patches merged 2 by 2, as Qwen2-VL reads them
        let other = ImageSizing {
            token_pixels: 28,
            min_pixels: 3_136,
            max_pixels: 1_000_000,
        };
        assert_eq!(image_size(&png(1000, 750), other).unwrap(), (1008, 756));
        assert_eq!(image_tokens(&png(1000, 750), other).unwrap(), 36 * 27);
        // over the maximum, the image shrinks below a million pixels
        let (width, height) = image_size(&png(4000, 3000), other).unwrap();
        assert!(width * height <= 1_000_000, "{width} by {height}");
    }

    #[test]
    fn a_spec_reads_images_only_when_it_loads_its_vision_tower() {
        let spec = serde_json::json!({"vision_tower": true, "image_patch_pixels": 16,
                                      "image_merge": 2, "image_min_pixels": 65_536,
                                      "image_max_pixels": 16_777_216});
        assert_eq!(ImageSizing::of_model(&spec), Some(QWEN));
        let mut partial = spec.clone();
        partial["checkpoint_layers"] = 24.into();
        assert_eq!(ImageSizing::of_model(&partial), None);
        let mut decision = spec;
        decision["role"] = "decision".into();
        assert_eq!(ImageSizing::of_model(&decision), None);
        assert_eq!(
            ImageSizing::of_model(&serde_json::json!({"vision_tower": false})),
            None
        );
    }

    #[test]
    fn an_image_is_resized_as_qwens_image_processor_resizes_it() {
        // expected sizes from Hugging Face's smart_resize with Qwen3.5's
        // preprocessor_config.json; a small image grows to 65,536 pixels
        for ((width, height), resized) in [
            ((1184, 1536), (1184, 1536)),
            ((1000, 750), (992, 736)),
            ((60, 96), (224, 352)),
            ((20000, 15000), (4704, 3520)),
            ((1040, 1040), (1024, 1024)),
        ] {
            assert_eq!(image_size(&png(width, height), QWEN).unwrap(), resized);
        }
        assert_eq!(image_size(&jpeg(1000, 750), QWEN).unwrap(), (992, 736));
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
