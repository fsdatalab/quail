//! `QuailExtension`: Quail added to an existing DataFusion session.
//!
//! A host builds its own `SessionState` and adds Quail to it:
//!
//! ```ignore
//! let state = SessionStateBuilder::new().with_default_features().build();
//! let state = QuailExtension::new(runner).register(state)?;
//! let context = SessionContext::new_with_state(state);
//! context.sql("SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('Negative? {0}', r.body))")
//! ```
//!
//! A host with its own `QueryPlanner` calls [`QuailExtension::register_rules`]
//! instead. Its planner calls [`crate::claim_model_regions`] on the
//! optimized plan and has [`QuailExtension::extension_planner`] among its
//! extension planners.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Result, plan_err};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionState;
use datafusion::execution::registry::FunctionRegistry;
use datafusion::logical_expr::{Extension, LogicalPlan, UserDefinedLogicalNode};
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_planner::ExtensionPlanner;

use crate::functions::{QuailFunction, quail_functions};
use crate::region::exec::{ModelRegionPlanner, QuailQueryPlanner, RegionRunner};
use crate::region::filter_guard::{BEFORE_RULE, GuardRegionFilters};
use crate::rules::{logical_rules, with_quail_rules};
use crate::{LowerAiCalls, SnowflakeRegexpLike, StructFieldAccess};

/// Quail's additions to a DataFusion session: the AI functions, the
/// planner of `value['field']` on a struct in Snowflake SQL, the rules
/// that lower AI calls, optimize around them, and claim model
/// regions, and the planner that runs each region on `runner`.
#[derive(Debug)]
pub struct QuailExtension {
    runner: Arc<dyn RegionRunner>,
    /// The optimizer rules, by name, or None for the session's own rules
    /// with Quail's among them.
    rules: Option<Vec<String>>,
}

impl QuailExtension {
    /// Quail with the session's optimizer rules and Quail's among them
    /// ([`crate::rules`]), running regions on `runner`.
    pub fn new(runner: Arc<dyn RegionRunner>) -> Self {
        Self {
            runner,
            rules: None,
        }
    }

    /// Run `rules`, names [`crate::logical_rule`] knows, in order, as the
    /// session's optimizer rules.
    ///
    /// # Errors
    ///
    /// A rule name is unknown.
    pub fn with_logical_rules(mut self, rules: Vec<String>) -> Result<Self> {
        logical_rules(&rules)?;
        self.rules = Some(rules);
        Ok(self)
    }

    /// Add Quail to `state`, with Quail's query planner in place of the
    /// host's.
    ///
    /// # Errors
    ///
    /// The host has its own function under one of Quail's names.
    pub fn register(&self, state: SessionState) -> Result<SessionState> {
        self.registered(state, true)
    }

    /// Add Quail's functions and rules to `state`, keeping its query
    /// planner; the host's planner calls [`crate::claim_model_regions`]
    /// and has [`Self::extension_planner`].
    ///
    /// `LowerAiCalls` runs first, on the plan as DataFusion's SQL planner
    /// bound it, before the host's analyzer rules such as type coercion.
    /// Quail's optimizer rules join the host's, as [`crate::rules`]
    /// explains, unless [`Self::with_logical_rules`] names the list. The
    /// host's session id, function rewrites, and other rules stay.
    /// `regexp_like` gains the names `rlike` and `regexp_contains`.
    ///
    /// # Errors
    ///
    /// The host has its own function under one of Quail's names.
    pub fn register_rules(&self, state: SessionState) -> Result<SessionState> {
        self.registered(state, false)
    }

    fn registered(&self, state: SessionState, planner: bool) -> Result<SessionState> {
        let functions = quail_functions();
        for function in &functions {
            if let Ok(found) = state.udf(function.name())
                && found.inner().downcast_ref::<QuailFunction>().is_none()
            {
                return plan_err!(
                    "the session already has a function named {}",
                    function.name()
                );
            }
        }
        let first = state
            .analyzer()
            .rules
            .iter()
            .all(|rule| rule.name() != LowerAiCalls.name());
        let optimizer = match &self.rules {
            Some(names) => logical_rules(names)?,
            None => with_quail_rules(state.optimizers()),
        };
        // a builder made from a state forgets its session id
        let session_id = state.session_id().to_string();
        let mut builder = SessionStateBuilder::new_from_existing(state).with_session_id(session_id);
        if let Some(analyzer) = builder.analyzer() {
            analyzer
                .rules
                .retain(|rule| rule.name() != LowerAiCalls.name());
            analyzer.rules.insert(0, Arc::new(LowerAiCalls));
        }
        if let Some(found) = builder.optimizer() {
            found.rules = optimizer;
        }
        if let Some(physical) = builder.physical_optimizers() {
            let physical = &mut physical.rules;
            // the guard names the columns a region's dynamic filters may
            // read, before DataFusion pushes those filters down
            let guard = Arc::new(GuardRegionFilters);
            physical.retain(|rule| rule.name() != guard.name());
            let position = physical
                .iter()
                .position(|rule| rule.name() == BEFORE_RULE)
                .unwrap_or(physical.len());
            physical.insert(position, guard);
        }
        if planner {
            builder = builder
                .with_query_planner(Arc::new(QuailQueryPlanner::new(Arc::clone(&self.runner))));
        }
        let mut state = builder.build();
        for function in functions {
            state.register_udf(function)?;
        }
        state.register_udf(crate::functions::regexps::regexp_functions())?;
        if first {
            state.register_expr_planner(Arc::new(StructFieldAccess))?;
            state.register_function_rewrite(Arc::new(SnowflakeRegexpLike))?;
        }
        Ok(state)
    }

    /// The `ExtensionPlanner` that turns each model region into a
    /// `ModelRegionExec`, for a host's own query planner.
    pub fn extension_planner(&self) -> Arc<dyn ExtensionPlanner + Send + Sync> {
        Arc::new(ModelRegionPlanner::new(Arc::clone(&self.runner)))
    }
}

/// `plan`'s operator over other inputs, its expressions as they are.
///
/// `LogicalPlan::with_new_exprs` cannot rebuild a `DISTINCT ON` with an
/// `ORDER BY`; this keeps the operator and replaces its inputs alone.
pub(crate) fn with_inputs(plan: &LogicalPlan, inputs: Vec<LogicalPlan>) -> Result<LogicalPlan> {
    let mut inputs = inputs.into_iter();
    plan.clone()
        .map_children(|input| Ok(Transformed::yes(inputs.next().unwrap_or(input))))?
        .data
        .recompute_schema()
}

/// A plan whose root is `node`, one of Quail's extension nodes.
pub(crate) fn extension_node(node: impl UserDefinedLogicalNode + 'static) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

/// The plan's root as the extension node `T`, if it is one.
pub(crate) fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(extension) => extension.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}
