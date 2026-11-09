//! The prefix statistics of a small hand-made document set.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, LargeListArray};
use arrow::datatypes::Int32Type;
use quail_inputs::{Documents, PrefixIndex, ReaderGroup, first_documents, last_tokens};

/// Two documents that share a prefix with a third, an exact duplicate,
/// an empty document, and a prefix of another document.
fn tokens() -> LargeListArray {
    let documents: [&[i32]; 6] = [
        &[1, 2, 3, 4, 5],
        &[1, 2, 3, 4, 9],
        &[1, 2, 3, 4, 5],
        &[],
        &[7, 8],
        &[1, 2],
    ];
    LargeListArray::from_iter_primitive::<Int32Type, _, _>(
        documents
            .iter()
            .map(|document| Some(document.iter().map(|token| Some(*token)))),
    )
}

const LENGTHS: [i64; 6] = [5, 5, 5, 0, 2, 2];

#[test]
fn documents_sort_by_tokens_and_share_all_but_their_last_token() {
    let batches = [tokens()];
    let index = PrefixIndex::new(&Documents::new(&batches));
    assert_eq!(index.order, vec![3, 5, 0, 2, 1, 4]);
    // document 2 equals document 0 but computes its last token
    assert_eq!(index.shared, vec![2, 4, 4, 0, 0, 0]);
}

#[test]
fn readers_of_one_prefix_group_under_one_parent() {
    let batches = [tokens()];
    let index = PrefixIndex::new(&Documents::new(&batches));

    let tree = index.tree(2);
    assert_eq!(
        tree.parent,
        vec![Some(5), Some(0), Some(0), None, None, None]
    );
    assert_eq!(
        tree.reader_groups(&LENGTHS),
        vec![
            ReaderGroup {
                shared: 2,
                count: 1,
                rows: 3
            },
            ReaderGroup {
                shared: 4,
                count: 2,
                rows: 2
            },
        ]
    );

    // at four tokens a page, document 0 borrows nothing from document 5
    let tree = index.tree(4);
    assert_eq!(tree.shared, vec![0, 4, 4, 0, 0, 0]);
    assert_eq!(tree.parent, vec![None, Some(0), Some(0), None, None, None]);
    assert_eq!(
        tree.reader_groups(&LENGTHS),
        vec![ReaderGroup {
            shared: 4,
            count: 2,
            rows: 2
        }]
    );
}

#[test]
fn last_tokens_are_the_distinct_ends_of_documents_with_tokens() {
    let batches = [tokens()];
    assert_eq!(last_tokens(&Documents::new(&batches)), vec![2, 5, 8, 9]);
}

#[test]
fn a_duplicate_with_another_value_is_its_own_first_document() {
    let batches = [tokens()];
    let documents = Documents::new(&batches);
    assert_eq!(
        first_documents(&documents, &[]).unwrap(),
        vec![0, 1, 0, 3, 4, 5]
    );
    let stars: ArrayRef = Arc::new(Int64Array::from(vec![5, 4, 1, 3, 2, 1]));
    assert_eq!(
        first_documents(&documents, &[stars]).unwrap(),
        vec![0, 1, 2, 3, 4, 5]
    );
}
