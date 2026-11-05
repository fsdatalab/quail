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
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_planner::ExtensionPlanner;

use crate::exec::{ModelRegionPlanner, QuailQueryPlanner, RegionRunner};
use crate::functions::quail_functions;
use crate::{ClaimModelRegions, LowerAiCalls, OptimizeModelRegions, region_rule};

/// Quail's additions to a DataFusion session: the AI functions, the
/// rules that claim, lower, and optimize model regions, and the planner
/// that runs each region on `runner`.
#[derive(Debug)]
pub struct QuailExtension {
    runner: Arc<dyn RegionRunner>,
    rules: Vec<String>,
}

impl QuailExtension {
    /// Quail with the default model region rules, running regions on
    /// `runner`.
    pub fn new(runner: Arc<dyn RegionRunner>) -> Self {
        Self {
            runner,
            rules: crate::REGION_RULES
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
    }

    /// Run `rules`, names [`region_rule`] knows, inside each model region.
    ///
    /// # Errors
    ///
    /// A rule name is unknown.
    pub fn with_region_rules(mut self, rules: Vec<String>) -> Result<Self> {
        for name in &rules {
            region_rule(name)?;
        }
        self.rules = rules;
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
    /// Quail's analyzer rules run first, on the plan as DataFusion's SQL
    /// planner bound it, before the host's rules such as type coercion.
    ///
    /// # Errors
    ///
    /// A Quail function's name is taken by one `state` cannot replace.
    pub fn register_rules(&self, state: SessionState) -> Result<SessionState> {
        let mut analyzer: Vec<Arc<dyn AnalyzerRule + Send + Sync>> =
            vec![Arc::new(ClaimModelRegions::new()), Arc::new(LowerAiCalls)];
        analyzer.extend(
            state
                .analyzer()
                .rules
                .iter()
                .filter(|rule| !matches!(rule.name(), "claim_model_regions" | "lower_ai_calls"))
                .cloned(),
        );
        let mut state = SessionStateBuilder::new_from_existing(state)
            .with_analyzer_rules(analyzer)
            .with_optimizer_rule(Arc::new(OptimizeModelRegions::new(self.rules.clone())))
            .build();
        for function in quail_functions() {
            state.register_udf(function)?;
        }
        Ok(state)
    }

    /// The `ExtensionPlanner` that turns each model region into a
    /// `ModelRegionExec`, for a host's own query planner.
    pub fn extension_planner(&self) -> Arc<dyn ExtensionPlanner + Send + Sync> {
        Arc::new(ModelRegionPlanner::new(Arc::clone(&self.runner)))
    }
}
