//! `QuailExtension`: Quail added to an existing DataFusion session.
//!
//! A host builds its own `SessionState` and adds Quail to it:
//!
//! ```ignore
//! let state = SessionStateBuilder::new().with_default_features().build();
//! let state = QuailExtension::new(runner).register(state)?;
//! let context = SessionContext::new_with_state(state);
//! context.sql("SELECT r.id FROM reviews r WHERE ai_filter(prompt('Negative? {0}', r.body))")
//! ```
//!
//! A host with its own `QueryPlanner` calls [`QuailExtension::register_rules`]
//! instead and adds [`QuailExtension::extension_planner`] to its planner's
//! extension planners.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionState;
use datafusion::execution::registry::FunctionRegistry;
use datafusion::logical_expr::{Extension, LogicalPlan, UserDefinedLogicalNode};
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_planner::ExtensionPlanner;

use crate::functions::quail_functions;
use crate::region::exec::{ModelRegionPlanner, QuailQueryPlanner, RegionRunner};
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
    /// session's optimizer rules; `ClaimModelRegions` runs after them.
    ///
    /// # Errors
    ///
    /// A rule name is unknown.
    pub fn with_logical_rules(mut self, rules: Vec<String>) -> Result<Self> {
        logical_rules(&rules)?;
        self.rules = Some(rules);
        Ok(self)
    }

    /// Add Quail to `state`, with Quail's query planner.
    ///
    /// # Errors
    ///
    /// A Quail function's name is taken by one `state` cannot replace.
    pub fn register(&self, state: SessionState) -> Result<SessionState> {
        let state = self.register_rules(state)?;
        Ok(SessionStateBuilder::new_from_existing(state)
            .with_query_planner(Arc::new(QuailQueryPlanner::new(Arc::clone(&self.runner))))
            .build())
    }

    /// Add Quail's functions and rules to `state`, keeping its query
    /// planner; the host adds [`Self::extension_planner`] to it.
    ///
    /// `LowerAiCalls` runs first, on the plan as DataFusion's SQL planner
    /// bound it, before the host's analyzer rules such as type coercion.
    /// Quail's optimizer rules join the host's, as [`crate::rules`]
    /// explains, unless [`Self::with_logical_rules`] names the list.
    ///
    /// # Errors
    ///
    /// A Quail function's name is taken by one `state` cannot replace.
    pub fn register_rules(&self, state: SessionState) -> Result<SessionState> {
        let mut analyzer: Vec<Arc<dyn AnalyzerRule + Send + Sync>> = vec![Arc::new(LowerAiCalls)];
        analyzer.extend(
            state
                .analyzer()
                .rules
                .iter()
                .filter(|rule| rule.name() != "lower_ai_calls")
                .cloned(),
        );
        let optimizer = match &self.rules {
            Some(names) => logical_rules(names)?,
            None => with_quail_rules(state.optimizers()),
        };
        let mut state = SessionStateBuilder::new_from_existing(state)
            .with_analyzer_rules(analyzer)
            .with_optimizer_rules(optimizer)
            .build();
        for function in quail_functions() {
            state.register_udf(function)?;
        }
        state.register_udf(crate::functions::regexps::regexp_functions())?;
        state.register_expr_planner(Arc::new(StructFieldAccess))?;
        state.register_function_rewrite(Arc::new(SnowflakeRegexpLike))?;
        Ok(state)
    }

    /// The `ExtensionPlanner` that turns each model region into a
    /// `ModelRegionExec`, for a host's own query planner.
    pub fn extension_planner(&self) -> Arc<dyn ExtensionPlanner + Send + Sync> {
        Arc::new(ModelRegionPlanner::new(Arc::clone(&self.runner)))
    }
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
