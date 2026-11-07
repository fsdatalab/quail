//! Lower and refine every alternative's operators, then select the
//! cheapest, as `quail/planner/optimize.py` does.

use serde_json::{Map, Value};

use crate::context::PlanningContext;
use crate::expressions::{CompileError, Refusal};
use crate::graph::{PhysicalPlan, Planned, plan_region};
use crate::operators::Operators;
use crate::physical_rules::apply_plan_rules;

/// The selected plan and the index of the alternative it lowers.
pub struct Selected {
    pub plan: PhysicalPlan,
    pub alternative: usize,
}

/// Plan each alternative, apply the physical rules, and select the
/// cheapest, as `search_plan`. Ties go to the earlier alternative.
///
/// Each alternative is a plan's operators and the rewrites that produced
/// the plan.
///
/// # Errors
///
/// A callback fails, or a plan is malformed.
pub fn search_plan(
    alternatives: &[(Operators, Vec<String>)],
    context: &PlanningContext,
    rules: &[String],
) -> Result<Result<Selected, Refusal>, CompileError> {
    let mut candidates: Vec<Planned> = Vec::new();
    for (found, _) in alternatives {
        let mut candidate_context = context.clone();
        candidate_context.operators = Some(found.clone());
        let planned = match plan_region(found, &candidate_context)? {
            Ok(plan) => apply_plan_rules(plan, rules, &candidate_context)?,
            Err(refusal) => Err(refusal),
        };
        candidates.push(planned);
    }
    if candidates.is_empty() {
        return Ok(Err(Refusal::new(
            format!(
                "backend {} could not produce a plan",
                crate::expressions::repr_str(&context.backend)
            ),
            "no_physical_plan",
            1.0,
            0.0,
            "plans",
        )));
    }
    let seconds = |planned: &Planned| match planned {
        Ok(plan) => plan.estimated_seconds,
        Err(_) => f64::INFINITY,
    };
    let mut selected = 0;
    for (index, candidate) in candidates.iter().enumerate() {
        if seconds(candidate) < seconds(&candidates[selected]) {
            selected = index;
        }
    }
    let comparison: Vec<Value> = candidates
        .iter()
        .zip(alternatives)
        .enumerate()
        .map(|(index, (candidate, (_, rules)))| {
            let mut entry = Map::new();
            entry.insert("rewrites".into(), Value::from(rules.clone()));
            entry.insert(
                "estimated_seconds".into(),
                match candidate {
                    Ok(plan) => Value::from(plan.estimated_seconds),
                    Err(_) => Value::Null,
                },
            );
            entry.insert(
                "refusal".into(),
                match candidate {
                    Ok(_) => Value::Null,
                    Err(refusal) => Value::from(refusal.constraint.clone()),
                },
            );
            entry.insert("selected".into(), Value::from(index == selected));
            Value::Object(entry)
        })
        .collect();
    let mut plan = match candidates.swap_remove(selected) {
        Ok(plan) => plan,
        Err(refusal) => return Ok(Err(refusal)),
    };
    plan.settings
        .insert("candidates".into(), Value::Array(comparison));
    Ok(Ok(Selected {
        plan,
        alternative: selected,
    }))
}
