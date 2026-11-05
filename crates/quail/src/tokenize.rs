//! A tokenizer that a Python callable implements, such as a test's.

use std::sync::Arc;

use arrow::array::{Int32Builder, LargeListArray, LargeListBuilder, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field};
use arrow::error::ArrowError;
use pyo3::prelude::*;
use quail_inputs::Tokenizer;

/// Tokenizes with a Python callable `text -> list of tokens`, whose
/// tokens are integer ids or words.
pub struct PythonTokenizer {
    encode: Py<PyAny>,
    name: String,
}

impl PythonTokenizer {
    /// Tokenize with `encode`, reported under `name`.
    pub fn new(encode: Py<PyAny>, name: impl Into<String>) -> Self {
        Self {
            encode,
            name: name.into(),
        }
    }
}

enum Tokens {
    Ids(Vec<i32>),
    Words(Vec<String>),
}

impl Tokenizer for PythonTokenizer {
    fn encode_batch(&self, texts: &StringArray) -> Result<LargeListArray, ArrowError> {
        let found = Python::attach(|py| {
            texts
                .iter()
                .map(|text| {
                    let tokens = self.encode.call1(py, (text.unwrap_or(""),))?;
                    let tokens = tokens.bind(py);
                    match tokens.extract::<Vec<i32>>() {
                        Ok(ids) => Ok(Tokens::Ids(ids)),
                        Err(_) => Ok(Tokens::Words(tokens.extract::<Vec<String>>()?)),
                    }
                })
                .collect::<PyResult<Vec<Tokens>>>()
        })
        .map_err(|error| ArrowError::ExternalError(Box::new(error)))?;
        let words = found
            .iter()
            .any(|tokens| matches!(tokens, Tokens::Words(found) if !found.is_empty()));
        if words {
            let mut builder = LargeListBuilder::new(StringBuilder::new())
                .with_field(Arc::new(Field::new_list_field(DataType::Utf8, true)));
            for tokens in &found {
                if let Tokens::Words(words) = tokens {
                    for word in words {
                        builder.values().append_value(word);
                    }
                }
                builder.append(true);
            }
            return Ok(builder.finish());
        }
        let mut builder = LargeListBuilder::new(Int32Builder::new());
        for tokens in &found {
            if let Tokens::Ids(ids) = tokens {
                builder.values().append_slice(ids);
            }
            builder.append(true);
        }
        Ok(builder.finish())
    }

    fn name(&self) -> &str {
        &self.name
    }
}
