use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, BinaryBuilder, Int32Builder, ListBuilder, StructBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion::common::{Result, ScalarValue, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_interpret::util::TransformExt;
use hayro::hayro_syntax::Pdf;
use hayro::hayro_syntax::page::Page;
use hayro::kurbo::Affine;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, Resources, TargetInit};
use hayro::{RenderCache, RenderSettings, render_into};

use rayon::prelude::*;

use super::quiet::quietly;

const SIDE_MULTIPLE: u16 = 32;

const DEFAULT_LONGEST_EDGE: u16 = 1536;

const MAX_LONGEST_EDGE: u16 = 8192;

type Pages = Vec<(usize, Option<Vec<u8>>)>;

#[derive(Debug, PartialEq, Eq, Hash)]
struct PdfPages {
    signature: Signature,
    aliases: Vec<String>,
}

impl ScalarUDFImpl for PdfPages {
    fn name(&self) -> &str {
        "ai_pdf_pages"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [] | [_, _, _, _, _, ..] => plan_err!(
                "AI.PDF_PAGES takes a binary column of PDFs, and optionally longest_edge, \
                 the first page, and the last page; it takes {} arguments",
                arg_types.len()
            ),
            [
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                ..,
            ] => Ok(pages_type()),
            [other, ..] => plan_err!("AI.PDF_PAGES reads a binary column of PDFs, not {other}"),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let arg_types: Vec<DataType> = args
            .arg_fields
            .iter()
            .map(|field| field.data_type().clone())
            .collect();
        let pages = self.return_type(&arg_types)?;
        if let Some(longest_edge) = args.scalar_arguments.get(1) {
            longest_edge_pixels(*longest_edge)?;
        }
        page_range(
            args.scalar_arguments.get(2).copied().flatten(),
            args.scalar_arguments.get(3).copied().flatten(),
        )?;
        Ok(Arc::new(Field::new(self.name(), pages, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let literal = |index: usize| match args.args.get(index) {
            Some(ColumnarValue::Scalar(literal)) => Some(literal),
            _ => None,
        };
        let Some(pdf) = args.args.first() else {
            return exec_err!("AI.PDF_PAGES takes a binary column of PDFs");
        };
        let longest_edge = if args.args.len() > 1 {
            longest_edge_pixels(literal(1))?
        } else {
            DEFAULT_LONGEST_EDGE
        };
        let pages = page_range(literal(2), literal(3))?;
        let pdfs = pdf.to_array(args.number_rows)?;
        Ok(ColumnarValue::Array(pdf_pages(&pdfs, longest_edge, pages)?))
    }
}

fn pages_type() -> DataType {
    DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(page_fields()),
        true,
    )))
}

fn page_fields() -> Fields {
    Fields::from(vec![
        Field::new("page", DataType::Int32, true),
        Field::new("image", DataType::Binary, true),
    ])
}

fn longest_edge_pixels(longest_edge: Option<&ScalarValue>) -> Result<u16> {
    let Some(literal) = longest_edge.filter(|value| value.data_type().is_integer()) else {
        return plan_err!("AI.PDF_PAGES takes longest_edge as an integer literal, in pixels");
    };
    let ScalarValue::Int64(Some(pixels)) = literal.cast_to(&DataType::Int64)? else {
        return plan_err!("AI.PDF_PAGES takes longest_edge as an integer literal, in pixels");
    };
    match u16::try_from(pixels) {
        Ok(pixels) if pixels > 0 && pixels % SIDE_MULTIPLE == 0 && pixels <= MAX_LONGEST_EDGE => {
            Ok(pixels)
        }
        _ => plan_err!(
            "AI.PDF_PAGES takes a longest_edge that is a multiple of {SIDE_MULTIPLE} from \
             {SIDE_MULTIPLE} to {MAX_LONGEST_EDGE} pixels, not {pixels}"
        ),
    }
}

fn page_range(
    first: Option<&ScalarValue>,
    last: Option<&ScalarValue>,
) -> Result<(usize, Option<usize>)> {
    let page = |value: Option<&ScalarValue>, name: &str| -> Result<Option<usize>> {
        let Some(value) = value else {
            return Ok(None);
        };
        match value.cast_to(&DataType::Int64)? {
            ScalarValue::Int64(Some(found)) if found >= 1 => Ok(Some(found as usize)),
            _ => plan_err!("AI.PDF_PAGES takes the {name} page as an integer literal from 1"),
        }
    };
    let first = page(first, "first")?.unwrap_or(1);
    let last = page(last, "last")?;
    if last.is_some_and(|last| last < first) {
        return plan_err!("AI.PDF_PAGES takes a last page no earlier than its first page");
    }
    Ok((first, last))
}

fn pdf_pages(
    pdfs: &ArrayRef,
    longest_edge: u16,
    range: (usize, Option<usize>),
) -> Result<ArrayRef> {
    let files: Vec<Option<&[u8]>> = match pdfs.data_type() {
        DataType::Binary => pdfs.as_binary::<i32>().iter().collect(),
        DataType::LargeBinary => pdfs.as_binary::<i64>().iter().collect(),
        DataType::BinaryView => pdfs.as_binary_view().iter().collect(),
        other => return exec_err!("AI.PDF_PAGES reads a binary column, not {other}"),
    };
    let rendered = page_images(&files, longest_edge, range);
    let mut pages = ListBuilder::new(StructBuilder::from_fields(page_fields(), 0));
    for images in rendered {
        let Some(images) = images else {
            pages.append_null();
            continue;
        };
        let page = pages.values();
        for (number, image) in images {
            page.field_builder::<Int32Builder>(0)
                .expect("page is the first field")
                .append_value(number as i32);
            page.field_builder::<BinaryBuilder>(1)
                .expect("image is the second field")
                .append_option(image);
            page.append(true);
        }
        pages.append(true);
    }
    Ok(Arc::new(pages.finish()))
}

fn page_images(
    files: &[Option<&[u8]>],
    longest_edge: u16,
    (first, last): (usize, Option<usize>),
) -> Vec<Option<Pages>> {
    let parsed: Vec<Option<(Pdf, usize)>> = files
        .par_iter()
        .map(|bytes| {
            let bytes = (*bytes)?;
            let pdf = quietly(|| Pdf::new(bytes.to_vec()).ok())??;
            let num_pages = quietly(|| pdf.pages().len())?;
            Some((pdf, num_pages))
        })
        .collect();
    let in_range = |num_pages: usize| {
        let end = last.map_or(num_pages, |last| last.min(num_pages));
        (first - 1)..end.max(first - 1)
    };
    let tasks: Vec<(usize, usize)> = parsed
        .iter()
        .enumerate()
        .filter_map(|(file, found)| found.as_ref().map(|(_, num_pages)| (file, *num_pages)))
        .flat_map(|(file, num_pages)| in_range(num_pages).map(move |index| (file, index)))
        .collect();
    let settings = InterpreterSettings::default();
    let images: Vec<Option<Vec<u8>>> = tasks
        .par_iter()
        .map_init(RenderCache::new, |cache, &(file, index)| {
            let (pdf, _) = parsed[file].as_ref().expect("a task's PDF parsed");
            quietly(|| page_image(&pdf.pages()[index], cache, &settings, longest_edge)).flatten()
        })
        .collect();
    let mut images = images.into_iter();
    parsed
        .iter()
        .map(|found| {
            let (_, num_pages) = found.as_ref()?;
            Some(
                in_range(*num_pages)
                    .map(|index| (index + 1, images.next().expect("one image per task")))
                    .collect(),
            )
        })
        .collect()
}

fn image_size(width_points: f32, height_points: f32, longest_edge: u16) -> Option<(u16, u16)> {
    let has_size = |points: f32| points.is_finite() && points > 0.0;
    if !has_size(width_points) || !has_size(height_points) {
        return None;
    }
    let multiple = f32::from(SIDE_MULTIPLE);
    let other = |short: f32, long: f32| {
        let pixels = (short / long * f32::from(longest_edge) / multiple).round() * multiple;
        (pixels as u16).max(SIDE_MULTIPLE)
    };
    Some(if width_points >= height_points {
        (longest_edge, other(height_points, width_points))
    } else {
        (other(width_points, height_points), longest_edge)
    })
}

fn page_image<'a>(
    page: &'a Page<'a>,
    cache: &RenderCache<'a>,
    settings: &InterpreterSettings,
    longest_edge: u16,
) -> Option<Vec<u8>> {
    let (width_points, height_points) = page.render_dimensions();
    let (width, height) = image_size(width_points, height_points, longest_edge)?;
    let (width_points, height_points) = (f64::from(width_points), f64::from(height_points));
    let scale = f64::min(
        f64::from(width) / width_points,
        f64::from(height) / height_points,
    );
    let margin = (
        (f64::from(width) - width_points * scale) / 2.0,
        (f64::from(height) - height_points * scale) / 2.0,
    );
    let transform =
        Affine::translate(margin) * Affine::scale(scale) * page.initial_transform(true).to_kurbo();
    let mut context = RenderContext::new(width, height);
    render_into(
        page,
        cache,
        settings,
        &RenderSettings::default(),
        &mut context,
        transform,
    );
    context.flush();
    let mut pixmap = Pixmap::new(width, height);
    context.render_with(
        &mut pixmap,
        &mut Resources::default(),
        RasterizerSettings {
            target_init: TargetInit::Clear(WHITE),
            ..Default::default()
        },
    );
    png_bytes(&pixmap)
}

fn png_bytes(pixmap: &Pixmap) -> Option<Vec<u8>> {
    let rgb: Vec<u8> = pixmap
        .data_as_u8_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|rgba| &rgba[..3])
        .copied()
        .collect();
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(
        &mut bytes,
        u32::from(pixmap.width()),
        u32::from(pixmap.height()),
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().ok()?;
    writer.write_image_data(&rgb).ok()?;
    writer.finish().ok()?;
    Some(bytes)
}

pub(crate) fn pdf_pages_function() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(PdfPages {
        signature: Signature::variadic_any(Volatility::Immutable),
        aliases: vec!["AI.PDF_PAGES".to_string(), "ai.pdf_pages".to_string()],
    }))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::BinaryArray;
    use datafusion::arrow::datatypes::Int32Type;

    use super::*;

    const EVERY_PAGE: (usize, Option<usize>) = (1, None);

    fn line(text: &str) -> String {
        format!("BT /F1 12 Tf 72 540 Td ({text}) Tj ET")
    }

    fn pdf(pages: &[&str], width: u32, height: u32, rotate: u32) -> Vec<u8> {
        let font = 3 + 2 * pages.len();
        let kids: Vec<String> = (0..pages.len())
            .map(|index| format!("{} 0 R", 3 + 2 * index))
            .collect();
        let mut objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {} >>",
                kids.join(" "),
                pages.len()
            ),
        ];
        for (index, content) in pages.iter().enumerate() {
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] \
                 /Rotate {rotate} /Contents {} 0 R \
                 /Resources << /Font << /F1 {font} 0 R >> >> >>",
                4 + 2 * index
            ));
            objects.push(format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ));
        }
        objects.push(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
             /Encoding /WinAnsiEncoding >>"
                .to_string(),
        );
        let mut out = "%PDF-1.4\n".to_string();
        let mut offsets = Vec::new();
        for (index, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out += &format!("{} 0 obj\n{body}\nendobj\n", index + 1);
        }
        let xref = out.len();
        out += &format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
        for offset in offsets {
            out += &format!("{offset:010} 00000 n \n");
        }
        out += &format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        );
        out.into_bytes()
    }

    fn decoded(image: &[u8]) -> (u32, u32, Vec<u8>) {
        let mut reader = png::Decoder::new(std::io::Cursor::new(image))
            .read_info()
            .unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels).unwrap();
        assert_eq!(
            (info.color_type, info.bit_depth),
            (png::ColorType::Rgb, png::BitDepth::Eight)
        );
        pixels.truncate(info.buffer_size());
        (info.width, info.height, pixels)
    }

    fn one_pdf(pdf: &[u8], longest_edge: u16, range: (usize, Option<usize>)) -> Option<Pages> {
        page_images(&[Some(pdf)], longest_edge, range)
            .pop()
            .unwrap()
    }

    fn sizes(pdf: &[u8], longest_edge: u16) -> Vec<(u32, u32)> {
        one_pdf(pdf, longest_edge, EVERY_PAGE)
            .unwrap()
            .iter()
            .map(|(_, image)| {
                let (width, height, _) = decoded(image.as_ref().unwrap());
                (width, height)
            })
            .collect()
    }

    #[test]
    fn a_pdf_has_one_numbered_image_for_each_page() {
        let lease = [&*line("Lease from March 2025."), &*line("Rent is $2,400.")];
        let pdfs: ArrayRef = Arc::new(BinaryArray::from_iter_values([
            pdf(&lease, 612, 792, 0),
            pdf(&[&line("Invoice for one desk.")], 612, 792, 0),
        ]));
        let pages = pdf_pages(&pdfs, 1536, EVERY_PAGE).unwrap();
        assert_eq!(pages.data_type(), &pages_type());
        let pages = pages.as_list::<i32>();
        let numbers = |row: usize| {
            let page = pages.value(row);
            let numbers = page.as_struct().column(0).as_primitive::<Int32Type>();
            numbers.values().to_vec()
        };
        assert_eq!((numbers(0), numbers(1)), (vec![1, 2], vec![1]));
        let first = pages.value(0);
        let images = first.as_struct().column(1).as_binary::<i32>();
        for image in images.iter() {
            let (width, height, _) = decoded(image.unwrap());
            assert_eq!((width, height), (1184, 1536));
        }
    }

    #[test]
    fn each_side_is_a_multiple_of_32_and_the_longest_side_is_the_argument() {
        let letter = pdf(&[&line("Lease from March 2025.")], 612, 792, 0);
        assert_eq!(sizes(&letter, 1024), [(800, 1024)]);
        assert_eq!(sizes(&letter, 2048), [(1568, 2048)]);
        assert_eq!(image_size(100.0, 2000.0, 1536), Some((64, 1536)));
        assert_eq!(image_size(1.0, 2000.0, 1536), Some((32, 1536)));
        assert_eq!(image_size(500.0, 500.0, 1024), Some((1024, 1024)));
        assert_eq!(image_size(0.0, 500.0, 1024), None);
    }

    #[test]
    fn a_landscape_page_has_a_landscape_image() {
        let landscape = pdf(&[&line("Invoice for one desk.")], 792, 612, 0);
        assert_eq!(sizes(&landscape, 1536), [(1536, 1184)]);
        let turned = pdf(&[&line("Invoice for one desk.")], 612, 792, 90);
        assert_eq!(sizes(&turned, 1536), [(1536, 1184)]);
    }

    #[test]
    fn a_page_is_drawn_on_white_with_its_text() {
        let images = one_pdf(
            &pdf(&[&line("Lease from March 2025.")], 612, 792, 0),
            1024,
            EVERY_PAGE,
        )
        .unwrap();
        let (_, _, pixels) = decoded(images[0].1.as_ref().unwrap());
        assert_eq!(pixels[..3], [255, 255, 255]);
        let dark = pixels
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|rgb| rgb[0] < 128)
            .count();
        assert!(dark > 100, "{dark} dark pixels");
    }

    #[test]
    fn a_page_narrower_than_its_image_has_white_margins() {
        let tall = pdf(&["0 0 1 rg 0 0 100 1000 re f"], 100, 1000, 0);
        let images = one_pdf(&tall, 1024, EVERY_PAGE).unwrap();
        let (width, height, pixels) = decoded(images[0].1.as_ref().unwrap());
        assert_eq!((width, height), (96, 1024));
        let pixel = |x: usize, y: usize| &pixels[3 * (y * 96 + x)..][..3];
        assert_eq!(pixel(48, 16), [255, 255, 255]);
        assert_eq!(pixel(48, 512), [0, 0, 255]);
        assert_eq!(pixel(48, 1008), [255, 255, 255]);
    }

    #[test]
    fn a_range_draws_only_its_pages_and_numbers_them_from_the_pdfs_start() {
        let three = pdf(&[&line("one"), &line("two"), &line("three")], 612, 792, 0);
        let numbers = |range| -> Vec<usize> {
            (one_pdf(&three, 64, range).unwrap().into_iter())
                .map(|(number, _)| number)
                .collect()
        };
        assert_eq!(numbers((2, Some(3))), [2, 3]);
        assert_eq!(numbers((3, Some(9))), [3]);
        assert_eq!(numbers((2, None)), [2, 3]);
    }

    #[test]
    fn pages_drawn_in_parallel_return_to_their_own_pdf_in_order() {
        let two = pdf(&[&line("one"), &line("two")], 612, 792, 0);
        let three = pdf(&[&line("one"), &line("two"), &line("three")], 792, 612, 0);
        let files = [
            Some(&*two),
            None,
            Some(b"not a pdf".as_slice()),
            Some(&*three),
        ];
        let drawn: Vec<Option<Vec<_>>> = page_images(&files, 256, EVERY_PAGE)
            .into_iter()
            .map(|pages| {
                let size = |image: Option<Vec<u8>>| {
                    let (width, height, _) = decoded(&image.unwrap());
                    (width, height)
                };
                Some(
                    pages?
                        .into_iter()
                        .map(|(number, image)| (number, size(image)))
                        .collect(),
                )
            })
            .collect();
        assert_eq!(
            drawn,
            [
                Some(vec![(1, (192, 256)), (2, (192, 256))]),
                None,
                None,
                Some(vec![(1, (256, 192)), (2, (256, 192)), (3, (256, 192))]),
            ]
        );
    }

    #[test]
    fn a_file_that_is_not_a_pdf_has_no_pages() {
        let pdfs: ArrayRef = Arc::new(BinaryArray::from(vec![Some(b"not a pdf".as_slice()), None]));
        let pages = pdf_pages(&pdfs, 1536, EVERY_PAGE).unwrap();
        assert_eq!(pages.null_count(), 2);
    }

    #[test]
    fn a_longest_edge_that_is_not_a_multiple_of_32_is_refused() {
        let pixels = |value: i64| longest_edge_pixels(Some(&ScalarValue::Int64(Some(value))));
        assert_eq!(pixels(1024).unwrap(), 1024);
        for refused in [0, -32, 1000, 8224] {
            let message = pixels(refused).unwrap_err().to_string();
            assert!(message.contains("multiple of 32"), "{message}");
        }
        let text = ScalarValue::Utf8(Some("1024".to_string()));
        assert!(longest_edge_pixels(Some(&text)).is_err());
        assert!(longest_edge_pixels(None).is_err());
    }
}
