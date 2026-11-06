//! Token stores: a document column's tokens in one Arrow IPC file.
//!
//! The file is the one Python's `quail.common.tokens.TokenStore` opens:
//! a `__quail_token_ids` column of token lists and a
//! `__quail_token_count` column of their lengths, and optionally a
//! `__quail_token_offsets` column of each token's character range, with
//! the schema metadata `quail.kind = token_store`.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{AsArray, Int64Array, LargeListArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;

use crate::tokenizer::HfTokenizer;

/// The column of each document's tokens.
pub const TOKEN_COLUMN: &str = "__quail_token_ids";
/// The column of each document's token count.
pub const COUNT_COLUMN: &str = "__quail_token_count";
/// The column of each document's tokens' character ranges.
pub const OFFSET_COLUMN: &str = "__quail_token_offsets";

fn store_schema(with_offsets: bool) -> SchemaRef {
    let metadata = HashMap::from([("quail.kind".to_string(), "token_store".to_string())]);
    let mut fields = vec![
        Field::new(
            TOKEN_COLUMN,
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Int32, true))),
            false,
        ),
        Field::new(COUNT_COLUMN, DataType::Int64, false),
    ];
    if with_offsets {
        let pair =
            DataType::FixedSizeList(Arc::new(Field::new_list_field(DataType::Int64, true)), 2);
        fields.push(Field::new(
            OFFSET_COLUMN,
            DataType::LargeList(Arc::new(Field::new_list_field(pair, true))),
            true,
        ));
    }
    Arc::new(Schema::new(fields).with_metadata(metadata))
}

/// Writes a token store file batch by batch.
pub struct TokenStoreWriter {
    path: PathBuf,
    writer: Option<FileWriter<File>>,
}

impl TokenStoreWriter {
    /// Start a token store at `path`.
    ///
    /// # Errors
    ///
    /// The file cannot be created.
    pub fn create(path: &Path) -> Result<Self, ArrowError> {
        Self::create_with(path, false)
    }

    /// Start a token store at `path` that also holds each token's
    /// character range, which `AI.EXTRACT` reads to place its answers.
    ///
    /// # Errors
    ///
    /// The file cannot be created.
    pub fn create_with_offsets(path: &Path) -> Result<Self, ArrowError> {
        Self::create_with(path, true)
    }

    fn create_with(path: &Path, with_offsets: bool) -> Result<Self, ArrowError> {
        let file = File::create(path)?;
        let writer = FileWriter::try_new(file, &store_schema(with_offsets))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: Some(writer),
        })
    }

    /// Append the tokens of some documents, int32 ids.
    ///
    /// # Errors
    ///
    /// The tokens are not int32 ids, or the write fails.
    pub fn write(&mut self, tokens: LargeListArray) -> Result<(), ArrowError> {
        self.write_with(tokens, None)
    }

    fn write_with(
        &mut self,
        tokens: LargeListArray,
        ranges: Option<LargeListArray>,
    ) -> Result<(), ArrowError> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| ArrowError::InvalidArgumentError("the store is finished".into()))?;
        let counts: Int64Array = tokens
            .offsets()
            .windows(2)
            .map(|pair| Some(pair[1] - pair[0]))
            .collect();
        let mut columns: Vec<arrow::array::ArrayRef> = vec![Arc::new(tokens), Arc::new(counts)];
        if writer.schema().fields().len() == 3 {
            let ranges = ranges.ok_or_else(|| {
                ArrowError::InvalidArgumentError("the store holds character ranges".into())
            })?;
            columns.push(Arc::new(ranges));
        }
        let batch = RecordBatch::try_new(Arc::clone(writer.schema()), columns)?;
        writer.write(&batch)
    }

    /// Tokenize `texts` and append their tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer or the write fails.
    pub fn write_texts(
        &mut self,
        tokenizer: &HfTokenizer,
        texts: &StringArray,
    ) -> Result<(), ArrowError> {
        let with_offsets = self
            .writer
            .as_ref()
            .is_some_and(|writer| writer.schema().fields().len() == 3);
        if with_offsets {
            let (tokens, ranges) = tokenizer.encode_batch_with_offsets(texts)?;
            self.write_with(tokens, Some(ranges))
        } else {
            self.write(tokenizer.encode_batch(texts)?)
        }
    }

    /// Close the file and open it as a token store.
    ///
    /// # Errors
    ///
    /// The file does not close or read back.
    pub fn finish(mut self) -> Result<TokenStore, ArrowError> {
        if let Some(mut writer) = self.writer.take() {
            writer.finish()?;
        }
        TokenStore::open(&self.path)
    }
}

impl Drop for TokenStoreWriter {
    fn drop(&mut self) {
        // a writer dropped before finish leaves no partial file behind
        if self.writer.take().is_some() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// A document column's tokens, read from a token store file.
#[derive(Debug, Clone)]
pub struct TokenStore {
    path: PathBuf,
    tokens: Vec<LargeListArray>,
    lengths: Vec<i64>,
}

impl TokenStore {
    /// Open the token store file at `path`.
    ///
    /// # Errors
    ///
    /// The file is not a token store.
    pub fn open(path: &Path) -> Result<Self, ArrowError> {
        let reader = FileReader::try_new(File::open(path)?, None)?;
        let schema = reader.schema();
        let token_index = schema.index_of(TOKEN_COLUMN)?;
        let mut tokens = Vec::new();
        let mut lengths = Vec::new();
        for batch in reader {
            let batch = batch?;
            let column = batch.column(token_index).as_list::<i64>().clone();
            lengths.extend(column.offsets().windows(2).map(|pair| pair[1] - pair[0]));
            tokens.push(column);
        }
        Ok(Self {
            path: path.to_path_buf(),
            tokens,
            lengths,
        })
    }

    /// The file the store reads.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Each document's token count.
    pub fn lengths(&self) -> &[i64] {
        &self.lengths
    }

    /// The number of documents.
    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    /// Whether the store holds no document.
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    /// The token batches as the file holds them.
    pub fn batches(&self) -> &[LargeListArray] {
        &self.tokens
    }

    /// Each document's tokens, as one array per document position.
    pub fn documents(&self) -> Documents<'_> {
        Documents::new(&self.tokens)
    }
}

/// Each document's token ids, by document position.
pub struct Documents<'a>(Vec<&'a [i32]>);

impl<'a> Documents<'a> {
    fn new(batches: &'a [LargeListArray]) -> Self {
        let mut documents = Vec::new();
        for batch in batches {
            let values = batch.values().as_primitive::<Int32Type>().values();
            for pair in batch.offsets().windows(2) {
                documents.push(&values[pair[0] as usize..pair[1] as usize]);
            }
        }
        Documents(documents)
    }

    /// The token ids of each document.
    pub fn ids(&self) -> &[&'a [i32]] {
        &self.0
    }

    /// The number of documents.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there is no document.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
