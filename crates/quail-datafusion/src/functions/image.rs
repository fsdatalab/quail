//! `AI.IMAGE(image, longest_edge)`: an image whose longer side is at
//! most `longest_edge` pixels.
//!
//! A vision model reads one token per patch of pixels, so a large photo
//! costs thousands of tokens. The function shrinks each PNG or JPEG on
//! the CPU, without a model, before the model reads it. DataFusion runs
//! it as it runs any other scalar function, so it shrinks only the rows
//! that pass the filters below it. For example:
//!
//! ```sql
//! SELECT r.id FROM receipts r
//! WHERE AI.IF(PROMPT('Is the total over 100,000 rupiah? {0}', AI.IMAGE(r.image, 1024)))
//! ```
//!
//! Snowflake SQL may write `AI_IMAGE(r.image, 1024)`.

use std::io::Cursor;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, AsArray, LargeBinaryBuilder};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, ScalarValue, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat};

use super::quiet::quietly;

/// The largest `longest_edge` a call may name, as `AI.PDF_PAGES` allows.
const MAX_LONGEST_EDGE: u32 = 8192;

/// The quality of a shrunk JPEG, from 1 to 100.
const JPEG_QUALITY: u8 = 90;

/// `AI.IMAGE`, which shrinks each image to a longest side.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ShrinkImage {
    signature: Signature,
    aliases: Vec<String>,
}

impl ScalarUDFImpl for ShrinkImage {
    fn name(&self) -> &str {
        "ai_image"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                longest_edge,
            ] if longest_edge.is_integer() => Ok(DataType::LargeBinary),
            [
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                _,
            ] => {
                plan_err!("AI.IMAGE takes longest_edge as an integer literal, in pixels")
            }
            [other, _] => plan_err!("AI.IMAGE reads a binary column of images, not {other}"),
            _ => plan_err!(
                "AI.IMAGE takes a binary column of images and longest_edge; it takes {} \
                 arguments",
                arg_types.len()
            ),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [image, longest_edge] = args.args.as_slice() else {
            return exec_err!("AI.IMAGE takes a binary column of images and longest_edge");
        };
        let ColumnarValue::Scalar(longest_edge) = longest_edge else {
            return plan_err!("AI.IMAGE takes longest_edge as an integer literal, in pixels");
        };
        let longest_edge = longest_edge_pixels(longest_edge)?;
        let images = image.to_array(args.number_rows)?;
        Ok(ColumnarValue::Array(shrunk_images(&images, longest_edge)?))
    }
}

/// The `longest_edge` argument in pixels, or an error unless it is from
/// 1 to [`MAX_LONGEST_EDGE`].
fn longest_edge_pixels(longest_edge: &ScalarValue) -> Result<u32> {
    let ScalarValue::Int64(Some(pixels)) = longest_edge.cast_to(&DataType::Int64)? else {
        return plan_err!("AI.IMAGE takes longest_edge as an integer literal, in pixels");
    };
    match u32::try_from(pixels) {
        Ok(pixels) if (1..=MAX_LONGEST_EDGE).contains(&pixels) => Ok(pixels),
        _ => plan_err!(
            "AI.IMAGE takes a longest_edge from 1 to {MAX_LONGEST_EDGE} pixels, not {pixels}"
        ),
    }
}

/// Each image of `images`, a binary array, shrunk to `longest_edge`:
/// NULL for a NULL image or one that is not a PNG or a JPEG. The images
/// are `LargeBinary`, whose 64-bit offsets let DataFusion join batches of
/// more than 2 GiB of images into one.
fn shrunk_images(images: &ArrayRef, longest_edge: u32) -> Result<ArrayRef> {
    let mut shrunk = LargeBinaryBuilder::new();
    let mut append =
        |bytes: Option<&[u8]>| shrunk.append_option(bytes.and_then(|b| shrink(b, longest_edge)));
    match images.data_type() {
        DataType::Binary => images.as_binary::<i32>().iter().for_each(&mut append),
        DataType::LargeBinary => images.as_binary::<i64>().iter().for_each(&mut append),
        DataType::BinaryView => images.as_binary_view().iter().for_each(&mut append),
        other => return exec_err!("AI.IMAGE reads a binary column, not {other}"),
    }
    Ok(Arc::new(shrunk.finish()))
}

/// One image whose longer side is at most `longest_edge`, in the image's
/// own format, or None for bytes that are not a PNG or a JPEG. An image
/// already small enough keeps its bytes.
fn shrink(bytes: &[u8], longest_edge: u32) -> Option<Vec<u8>> {
    let format = image::guess_format(bytes).ok()?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        return None;
    }
    let decoded = quietly(|| image::load_from_memory_with_format(bytes, format).ok())??;
    let (width, height) = (decoded.width(), decoded.height());
    if width.max(height) <= longest_edge {
        return Some(bytes.to_vec());
    }
    let scale = f64::from(longest_edge) / f64::from(width.max(height));
    let side = |pixels: u32| ((f64::from(pixels) * scale).round() as u32).max(1);
    // Qwen's image processor resizes with bicubic interpolation
    let resized = decoded.resize_exact(side(width), side(height), FilterType::CatmullRom);
    let mut encoded = Cursor::new(Vec::new());
    if format == ImageFormat::Jpeg {
        let encoder = JpegEncoder::new_with_quality(&mut encoded, JPEG_QUALITY);
        DynamicImage::ImageRgb8(resized.to_rgb8())
            .write_with_encoder(encoder)
            .ok()?;
    } else {
        resized.write_to(&mut encoded, ImageFormat::Png).ok()?;
    }
    Some(encoded.into_inner())
}

/// `AI.IMAGE` as a DataFusion UDF, also named `AI.IMAGE` and `ai.image`.
pub(crate) fn image_function() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(ShrinkImage {
        signature: Signature::any(2, Volatility::Immutable),
        aliases: vec!["AI.IMAGE".to_string(), "ai.image".to_string()],
    }))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::BinaryArray;
    use image::{GenericImageView, RgbImage};

    use super::*;

    fn image(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(width, height))
            .write_to(&mut bytes, format)
            .unwrap();
        bytes.into_inner()
    }

    fn size(bytes: &[u8]) -> (u32, u32) {
        image::load_from_memory(bytes).unwrap().dimensions()
    }

    #[test]
    fn a_large_image_shrinks_to_its_longest_edge_in_its_own_format() {
        let png = shrink(&image(2000, 1500, ImageFormat::Png), 1024).unwrap();
        assert_eq!(image::guess_format(&png).unwrap(), ImageFormat::Png);
        assert_eq!(size(&png), (1024, 768));
        let jpeg = shrink(&image(1500, 3000, ImageFormat::Jpeg), 1024).unwrap();
        assert_eq!(image::guess_format(&jpeg).unwrap(), ImageFormat::Jpeg);
        assert_eq!(size(&jpeg), (512, 1024));
    }

    #[test]
    fn a_small_image_keeps_its_bytes() {
        let small = image(300, 200, ImageFormat::Png);
        assert_eq!(shrink(&small, 1024).unwrap(), small);
    }

    #[test]
    fn bytes_that_are_not_a_png_or_a_jpeg_have_no_image() {
        let images: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"not an image".as_slice()),
            None,
            Some(b"GIF89a, an image of another format".as_slice()),
        ]));
        let shrunk = shrunk_images(&images, 64).unwrap();
        assert_eq!(shrunk.null_count(), 3);
    }

    #[test]
    fn a_longest_edge_outside_its_range_is_refused() {
        assert!(longest_edge_pixels(&ScalarValue::Int64(Some(0))).is_err());
        assert!(longest_edge_pixels(&ScalarValue::Int64(Some(9000))).is_err());
        assert_eq!(
            longest_edge_pixels(&ScalarValue::Int64(Some(1000))).unwrap(),
            1000
        );
    }
}
