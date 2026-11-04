//! `a JOIN b JOIN c ON <predicate over a, b, c>`, read left to right.
//!
//! The SQL parser reads a JOIN without a constraint followed by another
//! JOIN as `a JOIN (b JOIN c ON ...)`, so the ON sees only `b` and `c`.
//! Inner joins are associative, so [`flatten_joins`] writes such a nest
//! as a left-deep chain, `a CROSS JOIN b JOIN c ON ...`, whose ON sees
//! every table before it.

use sqlparser::ast::{
    Join, JoinConstraint, JoinOperator, Query, SetExpr, Statement, TableFactor, TableWithJoins,
};

fn is_inner(operator: &JoinOperator) -> bool {
    matches!(
        operator,
        JoinOperator::Join(_) | JoinOperator::Inner(_) | JoinOperator::CrossJoin(_)
    )
}

fn unconstrained(operator: &JoinOperator) -> bool {
    matches!(
        operator,
        JoinOperator::Join(JoinConstraint::None)
            | JoinOperator::Inner(JoinConstraint::None)
            | JoinOperator::CrossJoin(JoinConstraint::None)
    )
}

fn flatten_table(table: &mut TableWithJoins) {
    let mut joins = Vec::new();
    for join in table.joins.drain(..) {
        let Join {
            relation,
            global,
            join_operator,
        } = join;
        match relation {
            TableFactor::NestedJoin {
                mut table_with_joins,
                alias: None,
            } => {
                flatten_table(&mut table_with_joins);
                if unconstrained(&join_operator)
                    && table_with_joins
                        .joins
                        .iter()
                        .all(|inner| is_inner(&inner.join_operator))
                {
                    let nested = *table_with_joins;
                    joins.push(Join {
                        relation: nested.relation,
                        global,
                        join_operator: JoinOperator::CrossJoin(JoinConstraint::None),
                    });
                    joins.extend(nested.joins);
                } else {
                    joins.push(Join {
                        relation: TableFactor::NestedJoin {
                            table_with_joins,
                            alias: None,
                        },
                        global,
                        join_operator,
                    });
                }
            }
            relation => joins.push(Join {
                relation,
                global,
                join_operator,
            }),
        }
    }
    table.joins = joins;
}

fn flatten_query(query: &mut Query) {
    if let SetExpr::Select(select) = query.body.as_mut() {
        for table in &mut select.from {
            flatten_table(table);
        }
    }
}

/// Write each unconstrained JOIN over a nest of inner joins as a
/// left-deep chain, as the module explains.
pub fn flatten_joins(statement: &mut Statement) {
    if let Statement::Query(query) = statement {
        flatten_query(query);
    }
}

#[cfg(test)]
mod tests {
    use sqlparser::dialect::SnowflakeDialect;
    use sqlparser::parser::Parser;

    use super::*;

    fn flattened(sql: &str) -> String {
        let mut statement = Parser::parse_sql(&SnowflakeDialect {}, sql)
            .unwrap()
            .remove(0);
        flatten_joins(&mut statement);
        statement.to_string()
    }

    #[test]
    fn a_join_without_on_before_another_join_reads_left_to_right() {
        assert_eq!(
            flattened("SELECT 1 FROM a JOIN b JOIN c ON f(a.x, b.y, c.z)"),
            "SELECT 1 FROM a CROSS JOIN b JOIN c ON f(a.x, b.y, c.z)"
        );
        // an outer join keeps its nest, and so does a constrained one
        for kept in [
            "SELECT 1 FROM a JOIN b LEFT JOIN c ON b.y = c.z",
            "SELECT 1 FROM a JOIN (b JOIN c ON b.y = c.z) ON a.x = b.y",
        ] {
            let parsed = Parser::parse_sql(&SnowflakeDialect {}, kept)
                .unwrap()
                .remove(0);
            assert_eq!(flattened(kept), parsed.to_string());
        }
    }
}
