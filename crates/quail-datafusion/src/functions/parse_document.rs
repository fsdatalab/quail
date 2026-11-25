use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, AsArray, StringBuilder};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::quiet::quietly;

const PAGE_BREAK: &str = "\u{c}";

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

fn document_texts(documents: &ArrayRef) -> Result<ArrayRef> {
    let mut texts = StringBuilder::with_capacity(documents.len(), 0);
    let mut append = |bytes: Option<&[u8]>| texts.append_option(bytes.and_then(pdf_text));
    match documents.data_type() {
        DataType::Binary => documents.as_binary::<i32>().iter().for_each(&mut append),
        DataType::LargeBinary => documents.as_binary::<i64>().iter().for_each(&mut append),
        DataType::BinaryView => documents.as_binary_view().iter().for_each(&mut append),
        other => return exec_err!("AI.PARSE_DOCUMENT reads a binary column, not {other}"),
    }
    Ok(Arc::new(texts.finish()))
}

fn pdf_text(bytes: &[u8]) -> Option<String> {
    let pages = quietly(|| pdf_extract::extract_text_from_mem_by_pages(bytes))?.ok()?;
    let pages: Vec<&str> = pages.iter().map(|page| page.trim()).collect();
    Some(pages.join(PAGE_BREAK))
}

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
