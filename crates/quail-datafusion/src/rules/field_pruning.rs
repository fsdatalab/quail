use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, internal_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, LogicalPlan};

use crate::ai::AiNode;
use crate::functions::options::unfolded;
use crate::plan::{as_node, extension_node};

pub(crate) fn question_fields(call: &Expr) -> Option<Vec<String>> {
    let Expr::ScalarFunction(found) = call else {
        return None;
    };
    let Some(Expr::ScalarFunction(questions)) = found.args.get(1) else {
        return None;
    };
    if questions.name() != "named_struct" || questions.args.len() % 2 != 0 {
        return None;
    }
    questions
        .args
        .chunks(2)
        .map(|pair| match &pair[0] {
            Expr::Literal(value, _) => value.try_as_str().flatten().map(ToString::to_string),
            _ => None,
        })
        .collect()
}

pub(crate) fn with_fields(call: &Expr, kept: &[usize]) -> Expr {
    let Expr::ScalarFunction(found) = call else {
        unreachable!("an extraction call")
    };
    let Some(Expr::ScalarFunction(questions)) = found.args.get(1) else {
        unreachable!("a questions argument")
    };
    let pairs: Vec<Expr> = kept
        .iter()
        .flat_map(|index| questions.args[2 * index..2 * index + 2].to_vec())
        .collect();
    let mut args = found.args.clone();
    args[1] = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::clone(&questions.func), pairs));
    Expr::ScalarFunction(ScalarFunction::new_udf(Arc::clone(&found.func), args))
}

pub(crate) fn replace_call(plan: LogicalPlan, old: &Expr, new: &Expr) -> Result<LogicalPlan> {
    let (Expr::ScalarFunction(old), Expr::ScalarFunction(new)) = (old, new) else {
        return internal_err!("an AI call is a function call");
    };
    let extraction = old.name() == "ai_extract";
    let rewrite = |expr: Expr| -> Result<Expr> {
        expr.transform_down(|found| {
            let Expr::ScalarFunction(call) = &found else {
                return Ok(Transformed::no(found));
            };
            let span = extraction && call.name() == "ai_extract_span";
            if !(span || call.name() == old.name())
                || call
                    .args
                    .iter()
                    .cloned()
                    .map(unfolded)
                    .ne(old.args.iter().cloned())
            {
                return Ok(Transformed::no(found));
            }
            Ok(Transformed::new(
                Expr::ScalarFunction(ScalarFunction::new_udf(
                    Arc::clone(&call.func),
                    new.args.clone(),
                )),
                true,
                TreeNodeRecursion::Jump,
            ))
        })
        .map(|done| done.data)
    };
    plan.transform_up_with_subqueries(|node| {
        let Some(ai) = as_node::<AiNode>(&node) else {
            return node.recompute_schema().map(Transformed::yes);
        };
        let mut ai = ai.map_expressions(rewrite)?;
        if ai.function() == "ai_extract"
            && matches!(ai.call(), Expr::ScalarFunction(call) if call.args == new.args)
        {
            ai = ai.retyped_extraction()?;
        }
        Ok(Transformed::yes(extension_node(ai.checked()?)))
    })
    .map(|done| done.data)
}
