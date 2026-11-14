//! The placements of a region's column operators that physical planning
//! prices, read from its operators.
//!
//! A table that an AI join reads, other than one an `EXISTS` gate only
//! tests, may run its classifications, extractions, maps, and their
//! tests below the join, on each of its rows, or above it, on the rows
//! the join keeps. [`placements`] returns the operators as bound, then,
//! when it differs, the operators with each such table's column
//! operators and tests above the joins.
//!
//! For example, imagine `reviews r JOIN aspects a ON AI.IF(...)` with a
//! classification of `r.body` named `sentiment`. As bound it runs on
//! every review; the second placement runs it on the reviews the join
//! keeps, and names the rule that moved it, [`LIFT`].
//!
//! Above the joins, the operators run in a fixed order: each table in
//! scan order, and for each, the operators the query placed above the
//! joins, then the ones moved there, each in its own order.

use std::collections::BTreeSet;

use crate::expressions::asked_call;
use crate::expressions::call_aliases;
use crate::operators::{ColumnTest, Operators, SemanticColumn};

/// The name shown for a candidate whose column operators run above a join.
pub const LIFT: &str = "should_lift_column_operators";

/// A region's operators and the rewrites that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct Alternative {
    pub operators: Operators,
    pub rules: Vec<String>,
}

/// Where a column operator or test runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Among the joins: on a table no AI join moves, or a classification
    /// of the rows an inner AI join keeps.
    Fixed,
    /// Above every join, as the query placed it.
    Above,
    /// On its table, below the joins, moved above them.
    Moved,
}

/// The operators as bound, then the operators with the column operators
/// of each table an AI join reads above the joins, when that differs.
pub fn placements(found: &Operators) -> Vec<Alternative> {
    let mut alternatives = vec![Alternative {
        operators: found.clone(),
        rules: Vec::new(),
    }];
    if let Some(lifted) = lifted(found)
        && lifted != *found
    {
        alternatives.push(Alternative {
            operators: lifted,
            rules: vec![LIFT.to_string()],
        });
    }
    alternatives
}

/// The aliases whose column operators an AI join lets move above it:
/// each table an AI join reads, except one a gate tests and keeps no
/// KV of.
fn moving_aliases(found: &Operators) -> BTreeSet<String> {
    let mut fixed = BTreeSet::new();
    let mut joined = BTreeSet::new();
    for join in &found.joins {
        for alias in call_aliases(asked_call(&join.predicate)) {
            if join.semantics != "full" && Some(&alias) != join.anchor.as_ref() {
                fixed.insert(alias.clone());
            }
            joined.insert(alias);
        }
    }
    joined.difference(&fixed).cloned().collect()
}

/// Whether a classification of joined rows runs above every join: it
/// follows the last AI join, and that join holds every table.
fn above_every_join(found: &Operators, column: &SemanticColumn) -> bool {
    let read: BTreeSet<String> = call_aliases(&column.call).into_iter().collect();
    let Some(last) = found.joins.last() else {
        return true;
    };
    let joined: BTreeSet<String> = call_aliases(asked_call(&last.predicate))
        .into_iter()
        .collect();
    let mut held: BTreeSet<String> = last.outer_aliases.iter().cloned().collect();
    held.extend(joined.iter().cloned());
    joined == read && found.scans.iter().all(|scan| held.contains(&scan.alias))
}

/// The operators with each moving table's column operators and tests
/// above the joins; None for a region of one table.
fn lifted(found: &Operators) -> Option<Operators> {
    if found.scans.len() < 2 {
        return None;
    }
    let moving = moving_aliases(found);
    let any_join = !found.joins.is_empty();
    let place_column = |column: &SemanticColumn| -> Place {
        let read = call_aliases(&column.call);
        if read.len() > 1 {
            return if above_every_join(found, column) {
                Place::Above
            } else {
                Place::Fixed
            };
        }
        if column.above_joins {
            Place::Above
        } else if moving.contains(&column.alias()) {
            Place::Moved
        } else {
            Place::Fixed
        }
    };
    let aliases: Vec<String> = found.scans.iter().map(|scan| scan.alias.clone()).collect();
    // the operators above the joins: each table in scan order, its
    // operators the query placed above first
    let order = |places: Vec<(Place, String)>| -> Vec<usize> {
        let mut ordered: Vec<usize> = (0..places.len())
            .filter(|index| places[*index].0 == Place::Fixed)
            .collect();
        for alias in &aliases {
            for place in [Place::Above, Place::Moved] {
                ordered.extend(
                    (0..places.len()).filter(|index| places[*index] == (place, alias.clone())),
                );
            }
        }
        ordered
    };
    let mut lifted = found.clone();
    for list in [
        &mut lifted.classifies,
        &mut lifted.extracts,
        &mut lifted.sae_features,
        &mut lifted.maps,
    ] {
        let places: Vec<(Place, String)> = list
            .iter()
            .map(|column| (place_column(column), column.alias()))
            .collect();
        let ordered: Vec<SemanticColumn> = order(places.clone())
            .into_iter()
            .map(|index| {
                let mut column = list[index].clone();
                if places[index].0 != Place::Fixed {
                    column.above_joins = any_join;
                }
                column
            })
            .collect();
        *list = ordered;
    }
    // each table's tests, in the order they now run, its labels' first
    let mut tests: Vec<(String, Vec<(Place, ColumnTest)>)> = Vec::new();
    for (alias, found_tests) in &found.column_filters {
        let placed: Vec<(Place, ColumnTest)> = found_tests
            .iter()
            .map(|test| {
                let place = if !test.before_joins {
                    Place::Above
                } else if moving.contains(alias) {
                    Place::Moved
                } else {
                    Place::Fixed
                };
                (place, test.clone())
            })
            .collect();
        tests.push((alias.clone(), placed));
    }
    let first_fixed = |alias: &String| {
        found.filters.iter().any(|(found, _)| found == alias)
            || tests.iter().any(|(found, placed)| {
                found == alias && placed.iter().any(|(place, _)| *place == Place::Fixed)
            })
    };
    // a table is first met at its AI filters or a test among the joins,
    // in scan order, else at its first test above them
    let mut met: Vec<&String> = aliases.iter().filter(|alias| first_fixed(alias)).collect();
    met.extend(aliases.iter().filter(|alias| !first_fixed(alias)));
    lifted.column_filters = Vec::new();
    for alias in met {
        let Some((_, placed)) = tests.iter().find(|(found, _)| found == alias) else {
            continue;
        };
        let mut ordered: Vec<ColumnTest> = Vec::new();
        for place in [Place::Fixed, Place::Above, Place::Moved] {
            for (found_place, test) in placed {
                if *found_place == place {
                    let mut test = test.clone();
                    if place != Place::Fixed {
                        test.before_joins = !any_join;
                    }
                    ordered.push(test);
                }
            }
        }
        ordered.sort_by_key(|test| test.call.kind != "label");
        let asks = found
            .filters
            .iter()
            .find(|(found, _)| found == alias)
            .map_or(0, |(_, predicates)| predicates.len());
        for (index, test) in ordered.iter_mut().enumerate() {
            test.position = asks + index;
        }
        lifted.column_filters.push((alias.clone(), ordered));
    }
    Some(lifted)
}
