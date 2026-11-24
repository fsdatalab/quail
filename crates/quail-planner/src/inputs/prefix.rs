use std::collections::{BTreeSet, HashMap};

use arrow::array::ArrayRef;
use arrow::error::ArrowError;
use arrow::row::{RowConverter, SortField};

use crate::inputs::store::Documents;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixIndex {
    pub order: Vec<usize>,
    pub shared: Vec<i64>,
}

fn common_prefix<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| left.len().min(right.len()))
}

impl PrefixIndex {
    pub fn new(documents: &Documents<'_>) -> Self {
        let documents = documents.ids();
        let mut order: Vec<usize> = (0..documents.len()).collect();
        order.sort_by(|&a, &b| documents[a].cmp(documents[b]));
        let mut shared = vec![0i64; documents.len()];
        for pair in order.windows(2) {
            let (previous, index) = (documents[pair[0]], documents[pair[1]]);
            let common = common_prefix(previous, index);
            shared[pair[1]] = common.min(index.len().saturating_sub(1)) as i64;
        }
        PrefixIndex { order, shared }
    }

    pub fn tree(&self, page_tokens: i64) -> PrefixTree {
        let shared: Vec<i64> = self
            .shared
            .iter()
            .map(|value| value / page_tokens * page_tokens)
            .collect();
        let mut parent: Vec<Option<usize>> = vec![None; shared.len()];
        for pair in self.order.windows(2) {
            let (previous, index) = (pair[0], pair[1]);
            let pages = shared[index];
            if pages > 0 {
                let mut owner = previous;
                while let Some(up) = parent[owner] {
                    if shared[owner] < pages {
                        break;
                    }
                    owner = up;
                }
                parent[index] = Some(owner);
            }
        }
        PrefixTree {
            order: self.order.clone(),
            parent,
            shared,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixTree {
    pub order: Vec<usize>,
    pub parent: Vec<Option<usize>>,
    pub shared: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReaderGroup {
    pub shared: i64,
    pub count: i64,
    pub rows: i64,
}

impl PrefixTree {
    pub fn reader_groups(&self, lengths: &[i64]) -> Vec<ReaderGroup> {
        let mut positions: HashMap<(usize, i64), usize> = HashMap::new();
        let mut groups: Vec<ReaderGroup> = Vec::new();
        for (index, parent) in self.parent.iter().enumerate() {
            let Some(parent) = parent else { continue };
            let shared = self.shared[index];
            let position = *positions.entry((*parent, shared)).or_insert_with(|| {
                groups.push(ReaderGroup {
                    shared,
                    count: 0,
                    rows: 0,
                });
                groups.len() - 1
            });
            groups[position].count += 1;
            groups[position].rows += lengths[index] - shared;
        }
        groups
    }
}

pub fn first_documents(
    documents: &Documents<'_>,
    columns: &[ArrayRef],
) -> Result<Vec<usize>, ArrowError> {
    let rows = if columns.is_empty() {
        None
    } else {
        let fields = columns
            .iter()
            .map(|column| SortField::new(column.data_type().clone()))
            .collect();
        Some(RowConverter::new(fields)?.convert_columns(columns)?)
    };
    let mut seen: HashMap<(&[i32], Vec<u8>), usize> = HashMap::new();
    let mut first = Vec::with_capacity(documents.len());
    for (index, tokens) in documents.ids().iter().enumerate() {
        let values = rows
            .as_ref()
            .map(|rows| rows.row(index).as_ref().to_vec())
            .unwrap_or_default();
        let key = (*tokens, values);
        first.push(*seen.entry(key).or_insert(index));
    }
    Ok(first)
}

pub fn last_tokens(documents: &Documents<'_>) -> Vec<i32> {
    let ends: BTreeSet<i32> = documents
        .ids()
        .iter()
        .filter_map(|tokens| tokens.last().copied())
        .collect();
    ends.into_iter().collect()
}
