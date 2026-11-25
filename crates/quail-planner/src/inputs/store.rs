use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Int64Array, LargeListArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use arrow::error::ArrowError;

use crate::inputs::tokenizer::HfTokenizer;

pub const TOKEN_COLUMN: &str = "__quail_token_ids";
pub const COUNT_COLUMN: &str = "__quail_token_count";
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

pub struct TokenStoreWriter {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl TokenStoreWriter {
    pub fn new() -> Self {
        Self::with(false)
    }

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

    pub fn to_table(&self) -> (SchemaRef, Vec<RecordBatch>) {
        (Arc::clone(&self.schema), self.batches.clone())
    }

    pub fn lengths(&self) -> &[i64] {
        &self.lengths
    }

    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    pub fn documents(&self) -> Documents<'_> {
        Documents::new(&self.tokens)
    }
}

pub struct Documents<'a>(Vec<&'a [i32]>);

impl<'a> Documents<'a> {
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

    pub fn ids(&self) -> &[&'a [i32]] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
