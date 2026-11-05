//! The statistics the planner reads from a table's tokens: which
//! documents repeat another, which share a prefix whose KV they can
//! read, and which tokens end the documents.

use std::collections::{BTreeSet, HashMap};

use arrow::array::ArrayRef;
use arrow::error::ArrowError;
use arrow::row::{RowConverter, SortField};

use crate::store::Documents;

/// The documents in sorted token order, and the tokens each shares with
/// its predecessor in that order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixIndex {
    /// Document positions in sorted token order.
    pub order: Vec<usize>,
    /// Per document, the tokens it has in common with its predecessor,
    /// short of its last token, which every document computes.
    pub shared: Vec<i64>,
}

fn common_prefix<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| left.len().min(right.len()))
}

impl PrefixIndex {
    /// Sort `documents` and measure what each shares with its predecessor.
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

    /// The prefix tree whose shares round down to `page_tokens`.
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

/// Which document each document borrows its KV prefix from, with the
/// borrowed tokens rounded down to a whole page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixTree {
    /// Document positions in sorted token order; a parent comes before
    /// its children.
    pub order: Vec<usize>,
    /// Per document, the position it borrows from, or None.
    pub parent: Vec<Option<usize>>,
    /// Per document, the borrowed tokens, a multiple of the page size.
    pub shared: Vec<i64>,
}

/// One group of documents that read the same borrowed prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReaderGroup {
    /// The borrowed tokens.
    pub shared: i64,
    /// The documents in the group.
    pub count: i64,
    /// The tokens the documents compute beyond the borrowed prefix.
    pub rows: i64,
}

impl PrefixTree {
    /// Group the documents that borrow by their parent and borrowed
    /// tokens, in the order each group first appears.
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

/// Each document's first document with the same tokens and the same
/// values in each of `columns`; a document no earlier one equals is its
/// own first.
///
/// # Errors
///
/// A column's type has no row encoding.
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

/// The distinct last tokens of the documents that have tokens, in order.
pub fn last_tokens(documents: &Documents<'_>) -> Vec<i32> {
    let ends: BTreeSet<i32> = documents
        .ids()
        .iter()
        .filter_map(|tokens| tokens.last().copied())
        .collect();
    ends.into_iter().collect()
}
