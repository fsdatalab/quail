//! `AI.PARSE_DOCUMENT(document)`: the text of a PDF held as bytes.
//!
//! The function reads the text each page stores, with `pdf-extract`, on
//! the CPU and without a model. DataFusion runs it as it runs any other
//! scalar function, below the model regions that read its text, so it
//! parses only the rows that pass the filters below it, once for each
//! distinct call. For example, with `PDF` a binary column:
//!
//! ```sql
//! SELECT f.path,
//!        AI.CLASSIFY(AI.PARSE_DOCUMENT(f.pdf), ['lease', 'invoice']) AS kind
//! FROM filings f
//! WHERE f.year = 2025
//! ```
//!
//! Snowflake SQL may write `AI_PARSE_DOCUMENT(f.pdf)`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, AsArray, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use rayon::prelude::*;

use super::quiet::quietly;

/// What separates one page's text from the next: a form feed, as
/// `pdftotext` writes it.
const PAGE_BREAK: &str = "\u{c}";

/// `AI.PARSE_DOCUMENT`, which returns each PDF's text.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ParseDocument {
    signature: Signature,
    aliases: Vec<String>,
}

impl ScalarUDFImpl for ParseDocument {
    fn name(&self) -> &str {
        "ai_parse_document"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::Binary | DataType::LargeBinary | DataType::BinaryView] => Ok(DataType::Utf8),
            [other] => plan_err!("AI.PARSE_DOCUMENT reads a binary column of PDFs, not {other}"),
            _ => plan_err!(
                "AI.PARSE_DOCUMENT takes one argument, a binary column of PDFs; it takes {}",
                arg_types.len()
            ),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [document] = args.args.as_slice() else {
            return exec_err!("AI.PARSE_DOCUMENT takes one argument");
        };
        let documents = document.to_array(args.number_rows)?;
        Ok(ColumnarValue::Array(document_texts(&documents)?))
    }
}

/// The text of each PDF in `documents`, a binary array: NULL for a NULL
/// or unreadable PDF.
fn document_texts(documents: &ArrayRef) -> Result<ArrayRef> {
    let files: Vec<Option<&[u8]>> = match documents.data_type() {
        DataType::Binary => documents.as_binary::<i32>().iter().collect(),
        DataType::LargeBinary => documents.as_binary::<i64>().iter().collect(),
        DataType::BinaryView => documents.as_binary_view().iter().collect(),
        other => return exec_err!("AI.PARSE_DOCUMENT reads a binary column, not {other}"),
    };
    let texts: Vec<Option<String>> = files
        .par_iter()
        .map(|bytes| (*bytes).and_then(pdf_text))
        .collect();
    Ok(Arc::new(StringArray::from(texts)))
}

/// The text a PDF stores, its pages in order and each page trimmed, or
/// None when `pdf-extract` cannot read the file.
fn pdf_text(bytes: &[u8]) -> Option<String> {
    // pdf-extract panics on some malformed files: 3 of ParseBench's 2,037
    let pages = quietly(|| pdf_extract::extract_text_from_mem_by_pages(bytes))?.ok()?;
    let pages: Vec<&str> = pages.iter().map(|page| page.trim()).collect();
    Some(pages.join(PAGE_BREAK))
}

/// `AI.PARSE_DOCUMENT` as a DataFusion UDF, also named
/// `AI.PARSE_DOCUMENT` and `ai.parse_document`.
pub(crate) fn parse_document_function() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(ParseDocument {
        signature: Signature::variadic_any(Volatility::Immutable),
        aliases: vec![
            "AI.PARSE_DOCUMENT".to_string(),
            "ai.parse_document".to_string(),
        ],
    }))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{BinaryArray, StringArray};

    use super::*;

    #[test]
    fn a_file_that_is_not_a_pdf_has_no_text() {
        let documents: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"not a pdf".as_slice()),
            Some(b"%PDF-1.4\ntruncated".as_slice()),
            None,
        ]));
        let texts = document_texts(&documents).unwrap();
        assert_eq!(
            texts.as_string::<i32>(),
            &StringArray::from(vec![None::<&str>, None, None])
        );
    }
}
