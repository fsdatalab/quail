//! Token stores: a document column's tokens as Arrow record batches in
//! memory.
//!
//! The batches are the ones Python's `quail.common.tokens.TokenStore`
//! reads: a `__quail_token_ids` column of token lists and a
//! `__quail_token_count` column of their lengths, and optionally a
//! `__quail_token_offsets` column of each token's character range, with
//! the schema metadata `quail.kind = token_store`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Int64Array, LargeListArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use arrow::error::ArrowError;

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

/// Builds a token store batch by batch.
pub struct TokenStoreWriter {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl TokenStoreWriter {
    /// Start an empty token store.
    pub fn new() -> Self {
        Self::with(false)
    }

    /// Start an empty token store that also holds each token's character
    /// range, which `AI.EXTRACT` reads to place its answers.
    pub fn with_offsets() -> Self {
        Self::with(true)
    }

    fn with(with_offsets: bool) -> Self {
        Self {
            schema: store_schema(with_offsets),
            batches: Vec::new(),
        }
    }

    fn has_offsets(&self) -> bool {
        self.schema.fields().len() == 3
    }

    /// Append the tokens of some documents, int32 ids.
    ///
    /// # Errors
    ///
    /// The tokens are not int32 ids, or the store holds character ranges.
    pub fn write(&mut self, tokens: LargeListArray) -> Result<(), ArrowError> {
        self.write_with(tokens, None)
    }

    fn write_with(
        &mut self,
        tokens: LargeListArray,
        ranges: Option<LargeListArray>,
    ) -> Result<(), ArrowError> {
        let counts: Int64Array = tokens
            .offsets()
            .windows(2)
            .map(|pair| Some(pair[1] - pair[0]))
            .collect();
        let mut columns: Vec<ArrayRef> = vec![Arc::new(tokens), Arc::new(counts)];
        if self.has_offsets() {
            let ranges = ranges.ok_or_else(|| {
                ArrowError::InvalidArgumentError("the store holds character ranges".into())
            })?;
            columns.push(Arc::new(ranges));
        }
        self.batches
            .push(RecordBatch::try_new(Arc::clone(&self.schema), columns)?);
        Ok(())
    }

    /// Tokenize `texts` and append their tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer fails.
    pub fn write_texts(
        &mut self,
        tokenizer: &HfTokenizer,
        texts: &StringArray,
    ) -> Result<(), ArrowError> {
        if self.has_offsets() {
            let (tokens, ranges) = tokenizer.encode_batch_with_offsets(texts)?;
            self.write_with(tokens, Some(ranges))
        } else {
            self.write(tokenizer.encode_batch(texts)?)
        }
    }

    /// The token store of the batches written.
    pub fn finish(self) -> TokenStore {
        let tokens = self
            .batches
            .iter()
            .map(|batch| batch.column(0).as_list::<i64>().clone())
            .collect();
        TokenStore::new(self.schema, self.batches, tokens)
    }
}

impl Default for TokenStoreWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A document column's tokens, held as record batches.
#[derive(Debug, Clone)]
pub struct TokenStore {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    tokens: Vec<LargeListArray>,
    lengths: Vec<i64>,
}

impl TokenStore {
    fn new(schema: SchemaRef, batches: Vec<RecordBatch>, tokens: Vec<LargeListArray>) -> Self {
        let lengths = tokens
            .iter()
            .flat_map(|column| column.offsets().windows(2).map(|pair| pair[1] - pair[0]))
            .collect();
        Self {
            schema,
            batches,
            tokens,
            lengths,
        }
    }

    /// The store's schema: the token, count, and optional offset columns.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The store's record batches.
    pub fn record_batches(&self) -> &[RecordBatch] {
        &self.batches
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

    /// Each document's tokens, as one array per document position.
    pub fn documents(&self) -> Documents<'_> {
        Documents::new(&self.tokens)
    }
}

/// Each document's token ids, by document position.
pub struct Documents<'a>(Vec<&'a [i32]>);

impl<'a> Documents<'a> {
    /// The documents of `batches` of int32 token lists, in order.
    ///
    /// # Panics
    ///
    /// A batch's values are not int32.
    pub fn new(batches: &'a [LargeListArray]) -> Self {
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
