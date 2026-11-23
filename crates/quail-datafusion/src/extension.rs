use std::sync::Arc;

use datafusion::common::{Result, plan_err};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionState;
use datafusion::execution::registry::FunctionRegistry;
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_planner::ExtensionPlanner;

use crate::functions::{QuailFunction, quail_functions};
use crate::region::exec::{ModelRegionPlanner, QuailQueryPlanner, RegionRunner};
use crate::region::filter_guard::{BEFORE_RULE, GuardRegionFilters};
use crate::region::stream_filters::StreamRegionRows;
use crate::rules::{logical_rules, with_quail_rules};
use crate::{LowerAiCalls, SnowflakeRegexpLike, StructFieldAccess};

#[derive(Debug)]
pub struct QuailExtension {
    runner: Arc<dyn RegionRunner>,
    rules: Option<Vec<String>>,
}

impl QuailExtension {
    pub fn new(runner: Arc<dyn RegionRunner>) -> Self {
        Self {
            runner,
            rules: None,
        }
    }

    pub fn with_logical_rules(mut self, rules: Vec<String>) -> Result<Self> {
        logical_rules(&rules)?;
        self.rules = Some(rules);
        Ok(self)
    }

    pub fn register(&self, state: SessionState) -> Result<SessionState> {
        self.registered(state, true)
    }

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
            let guard = Arc::new(GuardRegionFilters);
            physical.retain(|rule| rule.name() != guard.name());
            let position = physical
                .iter()
                .position(|rule| rule.name() == BEFORE_RULE)
                .unwrap_or(physical.len());
            physical.insert(position, guard);
            let stream = Arc::new(StreamRegionRows);
            physical.retain(|rule| rule.name() != stream.name());
            physical.push(stream);
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
        state.register_udf(crate::functions::image::image_function())?;
        state.register_udf(crate::functions::parse_document::parse_document_function())?;
        state.register_udf(crate::functions::pdf_pages::pdf_pages_function())?;
        if first {
            state.register_expr_planner(Arc::new(StructFieldAccess))?;
            state.register_function_rewrite(Arc::new(SnowflakeRegexpLike))?;
            if !hosts_json_functions(&state)? {
                datafusion_functions_json::register_all(&mut state)?;
            }
        }
        Ok(state)
    }

    pub fn extension_planner(&self) -> Arc<dyn ExtensionPlanner + Send + Sync> {
        Arc::new(ModelRegionPlanner::new(Arc::clone(&self.runner)))
    }
}

fn hosts_json_functions(state: &SessionState) -> Result<bool> {
    let mut names = SessionStateBuilder::new().build();
    datafusion_functions_json::register_all(&mut names)?;
    Ok(names
        .scalar_functions()
        .keys()
        .any(|name| state.scalar_functions().contains_key(name)))
}
