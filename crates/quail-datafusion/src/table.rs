//! `RunnerTable`: a table the region runner reads itself.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{Constraints, Result};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

/// A table the [`crate::RegionRunner`] reads by name inside a model
/// region, such as a Python provider whose token counts Quail caches.
///
/// DataFusion scans the inner table for the operators outside model
/// regions. A model region that reads any other table receives that
/// table's rows as an input.
pub struct RunnerTable {
    inner: Arc<dyn TableProvider>,
}

impl RunnerTable {
    /// Mark `inner` as read by the region runner.
    pub fn new(inner: Arc<dyn TableProvider>) -> Self {
        Self { inner }
    }
}

impl fmt::Debug for RunnerTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunnerTable").finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for RunnerTable {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.inner.constraints()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn get_logical_plan(&self) -> Option<Cow<'_, LogicalPlan>> {
        self.inner.get_logical_plan()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
}
