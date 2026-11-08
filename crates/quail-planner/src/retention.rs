//! Planned anchor reuse at each execution boundary, as
//! `quail/planner/graph/retention.py`.

use std::collections::{BTreeMap, BTreeSet};

use quail_cost::joins::PricedJoin;
use serde_json::{Map, Value};

use crate::statistics::thin;

/// Group consecutive full joins that share an anchor: each group's
/// positions in the sequence.
pub fn group_sequence(sequence: &[(PricedJoin, String)]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, (spec, anchor)) in sequence.iter().enumerate() {
        if let Some(group) = groups.last_mut() {
            let last = &sequence[*group.last().expect("a member")].0;
            let group_anchor = &sequence[group[0]].1;
            if spec.semantics == "full" && last.semantics == "full" && group_anchor == anchor {
                group.push(index);
                continue;
            }
        }
        groups.push(vec![index]);
    }
    groups
}

/// The aliases whose filter sequence feeds the join anchored on it, as
/// `streamed_aliases`.
pub fn streamed_aliases(
    sequence: &[(PricedJoin, String)],
    groups: &[Vec<usize>],
    ask_aliases: &BTreeSet<String>,
    gpus: i64,
) -> BTreeSet<String> {
    let mut streamed = BTreeSet::new();
    let mut partner_before = BTreeSet::new();
    for group in groups {
        let anchor = &sequence[group[0]].1;
        if gpus == 1
            && ask_aliases.contains(anchor)
            && !streamed.contains(anchor)
            && !partner_before.contains(anchor)
        {
            streamed.insert(anchor.clone());
        }
        for member in group {
            for alias in &sequence[*member].0.aliases {
                if alias != anchor && !streamed.contains(alias) {
                    partner_before.insert(alias.clone());
                }
            }
        }
    }
    streamed
}

/// The next anchor use and conditional survival at each boundary, as
/// `schedule`: {"initial", "before", "after"}.
pub fn schedule(
    sequence: &[(PricedJoin, String)],
    expected_rows: &BTreeMap<String, f64>,
    group_ids: Option<&[String]>,
) -> Value {
    let groups = group_sequence(sequence);
    let group_ids: Vec<String> = match group_ids {
        Some(ids) => ids.to_vec(),
        None => (0..groups.len())
            .map(|index| format!("group:{index}"))
            .collect(),
    };
    let mut counts = vec![expected_rows.clone()];
    for group in &groups {
        let mut after = counts.last().expect("counts").clone();
        for member in group {
            thin(&mut after, &sequence[*member].0);
        }
        counts.push(after);
    }
    let mut uses: Vec<Value> = Vec::new();
    for boundary in 0..=groups.len() {
        let mut upcoming = Map::new();
        for (index, group) in groups.iter().enumerate().skip(boundary) {
            let alias = &sequence[group[0]].1;
            if upcoming.contains_key(alias) {
                continue;
            }
            let current = counts[boundary][alias];
            let probability = if current != 0.0 {
                (counts[index][alias] / current).min(1.0)
            } else {
                0.0
            };
            upcoming.insert(
                alias.clone(),
                Value::Array(vec![Value::from(probability), Value::from(index)]),
            );
        }
        uses.push(Value::Object(upcoming));
    }
    let named = |offset: usize| {
        Value::Object(
            (0..groups.len())
                .map(|index| (group_ids[index].clone(), uses[index + offset].clone()))
                .collect(),
        )
    };
    let mut out = Map::new();
    out.insert("initial".into(), uses[0].clone());
    out.insert("before".into(), named(0));
    out.insert("after".into(), named(1));
    Value::Object(out)
}
