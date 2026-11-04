//! The label trie a classification's scoring reads, as
//! `quail/labels.py` builds it.

use std::collections::{BTreeMap, BTreeSet};

/// A node of the label trie: the label tokens from the root.
pub type Node = Vec<i64>;

pub const LETTERS_SCORING: &str = "letters";
pub const TREE_SCORING: &str = "trie_tree";
pub const DECODE_SCORING: &str = "trie_decode";

/// Each proper label prefix and the sorted tokens that can follow it.
///
/// # Errors
///
/// A label has no tokens.
pub fn label_trie(label_ids: &[Vec<i64>]) -> Result<BTreeMap<Node, Vec<i64>>, String> {
    let mut children: BTreeMap<Node, BTreeSet<i64>> = BTreeMap::new();
    for ids in label_ids {
        if ids.is_empty() {
            return Err("a label has no tokens".to_string());
        }
        for depth in 0..ids.len() {
            children
                .entry(ids[..depth].to_vec())
                .or_default()
                .insert(ids[depth]);
        }
    }
    Ok(children
        .into_iter()
        .map(|(node, tokens)| (node, tokens.into_iter().collect()))
        .collect())
}

/// The trie nodes whose rows the trie_tree rule reads.
///
/// # Errors
///
/// A label has no tokens.
pub fn read_nodes(label_ids: &[Vec<i64>], starts: &[Vec<bool>]) -> Result<BTreeSet<Node>, String> {
    let trie = label_trie(label_ids)?;
    let ends: BTreeSet<Node> = label_ids.iter().cloned().collect();
    let mut read: BTreeSet<Node> = BTreeSet::from([Vec::new()]);
    for (index, ids) in label_ids.iter().enumerate() {
        for depth in 1..ids.len() {
            let node = ids[..depth].to_vec();
            let starts_word = starts.get(index).map(|flags| flags[depth]).unwrap_or(false);
            if trie[&node].len() > 1
                || ends.contains(&node)
                || (read.contains(&ids[..depth - 1]) && !starts_word)
            {
                read.insert(node);
            }
        }
    }
    Ok(read)
}

/// The read nodes and their ancestors, whose tokens are fed.
pub fn fed_nodes(read: &BTreeSet<Node>) -> BTreeSet<Node> {
    read.iter()
        .flat_map(|node| (0..=node.len()).map(|depth| node[..depth].to_vec()))
        .collect()
}

/// The tokens a greedy decode feeds for each label, per round; the
/// first round's count includes the answer cue.
///
/// # Errors
///
/// A label has no tokens.
pub fn decode_round_tokens(label_ids: &[Vec<i64>]) -> Result<Vec<Vec<i64>>, String> {
    let trie = label_trie(label_ids)?;
    let mut choices: Vec<&Node> = trie
        .iter()
        .filter(|(_, tokens)| tokens.len() > 1)
        .map(|(node, _)| node)
        .collect();
    choices.sort_by_key(|node| (node.len(), (*node).clone()));
    let follow = |mut node: Node| {
        while let Some(tokens) = trie.get(&node) {
            if tokens.len() > 1 {
                break;
            }
            node.push(tokens[0]);
        }
        node
    };
    let first = follow(Vec::new());
    let mut run_lengths = vec![if trie.contains_key(&first) {
        first.len() as i64
    } else {
        0
    }];
    let mut after: BTreeMap<(Node, i64), (Node, Option<usize>)> = BTreeMap::new();
    for node in choices {
        for token in &trie[node] {
            let mut child = node.clone();
            child.push(*token);
            let target = follow(child);
            let run = if trie.contains_key(&target) {
                run_lengths.push((target.len() - node.len()) as i64);
                Some(run_lengths.len() - 1)
            } else {
                None
            };
            after.insert((node.clone(), *token), (target, run));
        }
    }
    Ok(label_ids
        .iter()
        .map(|ids| {
            let mut node = first.clone();
            let mut fed = vec![1 + run_lengths[0]];
            while trie.contains_key(&node) {
                let (target, run) = &after[&(node.clone(), ids[node.len()])];
                let Some(run) = run else { break };
                fed.push(run_lengths[*run]);
                node = target.clone();
            }
            fed
        })
        .collect())
}
